using System.ClientModel;
using System.Diagnostics;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using Microsoft.AI.Foundry.Local;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntimeGenAI;
using OpenAI;
using Windows.Security.Cryptography.Core;
using static Betalgo.Ranul.OpenAI.ObjectModels.Models;
using static Betalgo.Ranul.OpenAI.ObjectModels.RealtimeModels.RealtimeEventTypes;

// Config
int concurrency = 1; // adjust to run multiple requests in parallel later
string jsonlPath = "benchmarks.jsonl";
string csvPath = "benchmarks.csv";
string agentEndpoint = ""; // e.g., http://localhost:5280
string agentApiKey; // optional if needed

var httpClient = new HttpClient();

// You can parameterize this later
var models = new[]
{
    "phi-3.5-mini",
    "mistral-7b-v0.2",
    "qwen2.5-1.5b",
    "deepseek-r1-14b",
    "gpt-oss-20b"
};

var tasks = new[]
{
    ("reasoning", "You are a senior cloud architect. Explain this setup: an event-driven microservices system using Azure Service Bus + Azure Functions + Cosmos DB. Identify 3 scalability risks and 3 cost optimizations."),
    ("reasoning-Math", "Calculate the integral of x^2 from 0 to 1."),
    ("coding", "You are a senior C# engineer. Write a C# method that validates an IBAN (basic format check is enough) and add 5 xUnit tests that cover valid and invalid edge cases."),
    ("kpi", "You are a SaaS product analyst. Given this data:\n\nMonth,MRR,Churn,NPS\nJan,100000,4.1,45\nFeb,102500,3.9,48\nMar,105000,4.7,38\n\nSummarize the health of the product, and list the top 2 risks.")
};

var loggerFactory = LoggerFactory.Create(builder =>
{
    builder.SetMinimumLevel(Microsoft.Extensions.Logging.LogLevel.Information);
});

var logger = loggerFactory.CreateLogger("Bench");

var results = new List<BenchResult>();
var latencies = new Dictionary<string, List<long>>(); // key: model|task

// prepare output files
if (File.Exists(jsonlPath)) File.Delete(jsonlPath);
if (File.Exists(csvPath)) File.Delete(csvPath);
await File.WriteAllTextAsync(csvPath, "model,task,ms,chars,tokens,tokens_per_sec,cpu_pct,working_set_mb,gpu_name,gpu_mem_used_mb,gpu_mem_total_mb,gpu_util_pct,response\n");

// NVIDIA metrics helper
(bool ok, string name, double memUsedMb, double memTotalMb, double utilPct) GetNvidiaMetrics()
{
    try
    {
        var psi = new ProcessStartInfo
        {
            FileName = "nvidia-smi",
            Arguments = "--query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits",
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            UseShellExecute = false,
            CreateNoWindow = true
        };
        using var p = Process.Start(psi);
        var output = p!.StandardOutput.ReadToEnd();
        p.WaitForExit(3000);
        var line = output.Split('\n', StringSplitOptions.RemoveEmptyEntries).FirstOrDefault();
        if (string.IsNullOrWhiteSpace(line)) return (false, "", 0, 0, 0);
        var parts = line.Split(',', StringSplitOptions.TrimEntries);
        if (parts.Length < 4) return (false, "", 0, 0, 0);
        var name = parts[0];
        double.TryParse(parts[1], out var used);
        double.TryParse(parts[2], out var total);
        double.TryParse(parts[3], out var util);
        return (true, name, used, total, util);
    }
    catch
    {
        return (false, "", 0, 0, 0);
    }
}

// Local helpers
int EstimateTokens(ReadOnlySpan<char> text)
{
    int tokens = 0;
    bool inToken = false;
    for (int i = 0; i < text.Length; i++)
    {
        char c = text[i];
        if (!char.IsWhiteSpace(c))
        {
            if (!inToken)
            {
                tokens++;
                inToken = true;
            }
        }
        else
        {
            inToken = false;
        }
    }
    return tokens;
}

async Task<(OpenAIClient, Microsoft.AI.Foundry.Local.Model)> TryInvokeAgentAsync(string modelAlias)
{
    var mgr = await FoundryManagerSingleton.GetAsync();
    try
    {
        var catalog = await mgr.GetCatalogAsync();
        var model = await catalog.GetModelAsync(modelAlias) ?? throw new Exception("Model not found");
        await model.DownloadAsync(p =>
        {
            Console.Write($"\rDownloading model: {p:F2}%");
            if (p >= 100f) Console.WriteLine();
        });

        await model.LoadAsync();
        await mgr.StartWebServiceAsync();

        agentEndpoint = mgr.Urls.FirstOrDefault() ?? agentEndpoint;

        if (string.IsNullOrWhiteSpace(agentEndpoint))
        {
            throw new Exception($"[Error] No agent endpoint available.");
        }

        var url = new Uri(new Uri(agentEndpoint), "/v1");
        var key = new ApiKeyCredential("notneeded");
        OpenAIClient client = new OpenAIClient(key, new OpenAIClientOptions
        {
            Endpoint = new Uri(mgr.Urls.FirstOrDefault() + "/v1"),
        });

        return (client, model);
    }
    catch (Exception ex)
    {
        throw new Exception($"[Agent error] {ex.GetType().Name}: {ex.Message}");
    }
}

async Task<string> GetChat(OpenAIClient client, Microsoft.AI.Foundry.Local.Model model, string prompt)
{
    var chatClient = client.GetChatClient(model.Id);

    // Prefer non-streaming to avoid HttpIOException: Response ended prematurely
    try
    {
        var completion = chatClient.CompleteChat(prompt);
        var comp = completion.Value;
        string text = string.Empty;

        // Prefer comp.Content (list of content parts)
        if (comp?.Content != null && comp.Content.Count > 0)
        {
            text = comp.Content[0].Text ?? string.Empty;
        }
        else
        {
            // Fallback: serialize result and attempt to extract message content
            var raw = JsonSerializer.Serialize(comp);
            try
            {
                using var doc = JsonDocument.Parse(raw);
                if (doc.RootElement.TryGetProperty("content", out var contentArr) && contentArr.ValueKind == JsonValueKind.Array && contentArr.GetArrayLength() > 0)
                {
                    var first = contentArr[0];
                    if (first.TryGetProperty("text", out var t))
                    {
                        text = t.GetString() ?? string.Empty;
                    }
                }
            }
            catch
            {
                // ignore parse fallback errors
            }
        }

        Console.WriteLine($"[USER]: {prompt}\n\n[ASSISTANT]: {text}\n");
        return text;
    }
    catch (Exception ex)
    {
        // Fallback to streaming with defensive accumulation
        var completionUpdates = chatClient.CompleteChatStreaming(prompt);
        var body = new StringBuilder();
        Console.WriteLine($"[USER]: {prompt}\n");
        Console.Write("[ASSISTANT]: ");
        try
        {
            foreach (var update in completionUpdates)
            {
                if (update.ContentUpdate.Count > 0)
                {
                    var chunk = update.ContentUpdate[0].Text ?? string.Empty;
                    body.Append(chunk);
                    Console.Write(chunk);
                }
            }
        }
        catch (HttpIOException)
        {
            // Partial response collected; print a notice and continue
            Console.WriteLine("\n[Warning] Stream ended prematurely, using partial response.");
        }
        Console.WriteLine();
        return body.ToString();
    }
}

foreach (var (taskName, prompt) in tasks)
{
    Console.WriteLine();
    Console.WriteLine($"=== Task: {taskName} ===");

    foreach (var alias in models)
    {
        Console.WriteLine($"\nRunning {alias} …");

        var proc = Process.GetCurrentProcess();
        var cpuStart = proc.TotalProcessorTime;

        var (client, model) = await TryInvokeAgentAsync(alias);

        // Capture GPU metrics before inference
        var gpuBefore = GetNvidiaMetrics();

        var sw = Stopwatch.StartNew();

        // Simple retry with exponential backoff for transient errors
        string responseText = string.Empty;
        int maxRetries = 3;
        int attempt = 0;
        for (; attempt < maxRetries; attempt++)
        {
            try
            {
                responseText = await GetChat(client, model, prompt);
                break;
            }
            catch (Exception ex)
            {
                var delayMs = (int)Math.Min(2000, 250 * Math.Pow(2, attempt));
                Console.WriteLine($"[Retry {attempt + 1}/{maxRetries}] {ex.GetType().Name}: {ex.Message}. Waiting {delayMs} ms …");
                await Task.Delay(delayMs);
            }
        }

        sw.Stop();

        proc.Refresh();
        var cpuEnd = proc.TotalProcessorTime;

        // Capture GPU metrics after inference
        var gpuAfter = GetNvidiaMetrics();

        var content = responseText;
        var ms = sw.ElapsedMilliseconds;
        if (ms == 0) ms = 1; // avoid divide-by-zero leading to Infinity in metrics
        var chars = content.Length;
        var tokens = EstimateTokens(content);
        var elapsedSeconds = Math.Max(sw.Elapsed.TotalSeconds, 0.000001);
        var tokensPerSec = tokens / elapsedSeconds;

        // CPU percent over elapsed window (normalized by logical cores)
        var cpuMs = (cpuEnd - cpuStart).TotalMilliseconds;
        var cpuPct = cpuMs / ms / Environment.ProcessorCount * 100.0;
        var workingSetMb = proc.WorkingSet64 / (1024.0 * 1024.0);

        // sanitize metrics for JSON (no NaN/Infinity)
        if (double.IsNaN(tokensPerSec) || double.IsInfinity(tokensPerSec)) tokensPerSec = 0;
        if (double.IsNaN(cpuPct) || double.IsInfinity(cpuPct)) cpuPct = 0;
        if (double.IsNaN(workingSetMb) || double.IsInfinity(workingSetMb)) workingSetMb = 0;

        var result = new BenchResult(alias, taskName, ms, chars);
        results.Add(result);

        // accumulate latencies for percentiles
        var key = $"{alias}|{taskName}";
        if (!latencies.TryGetValue(key, out var list))
        {
            list = new List<long>();
            latencies[key] = list;
        }
        list.Add(ms);

        // structured JSONL per request
        var json = JsonSerializer.Serialize(new
        {
            timestamp = DateTimeOffset.UtcNow,
            model = alias,
            task = taskName,
            ms,
            chars,
            tokens,
            tokens_per_sec = Math.Round(tokensPerSec, 2),
            cpu_pct = Math.Round(cpuPct, 2),
            working_set_mb = Math.Round(workingSetMb, 2),
            prompt_len = prompt.Length,
            run_id = Guid.NewGuid().ToString("n"),
            response = content,
            retries = attempt,
            gpu_name = gpuAfter.name,
            gpu_mem_used_mb = Math.Round(gpuAfter.memUsedMb, 2),
            gpu_mem_total_mb = Math.Round(gpuAfter.memTotalMb, 2),
            gpu_util_pct = Math.Round(gpuAfter.utilPct, 2),
            gpu_before_util_pct = Math.Round(gpuBefore.utilPct, 2)
        });
        await File.AppendAllTextAsync(jsonlPath, json + "\n");

        // CSV row (escape quotes/newlines for Excel)
        var responseCsv = content.Replace("\"", "\"\"").Replace("\r", " ").Replace("\n", " ");
        await File.AppendAllTextAsync(csvPath, $"{alias},{taskName},{ms},{chars},{tokens},{tokensPerSec:F2},{cpuPct:F2},{workingSetMb:F2},\"{gpuAfter.name}\",{gpuAfter.memUsedMb:F2},{gpuAfter.memTotalMb:F2},{gpuAfter.utilPct:F2},\"{responseCsv}\"\n");

        logger.LogInformation("{Alias} | {Task} | {Ms} ms | {Chars} chars | {Tokens} tok | {TokPerSec} tok/s | CPU {CpuPct}% | WS {WsMb} MB | GPU {GpuName} util {GpuUtil}% mem {GpuMemUsed}/{GpuMemTotal} MB | Retries {Retries}",
                              alias, taskName, ms, chars, tokens, tokensPerSec, cpuPct, workingSetMb, gpuAfter.name, gpuAfter.utilPct, gpuAfter.memUsedMb, gpuAfter.memTotalMb, attempt);

        var snippet = content.Length > 300 ? content[..300] + "…" : content;
        Console.WriteLine($"> {alias} snippet:\n{snippet}\n");
    }
}

// Dump summary table at the end
Console.WriteLine("\n=== Summary ===");
foreach (var g in results.GroupBy(r => r.Task))
{
    Console.WriteLine($"\nTask: {g.Key}");
    foreach (var r in g.OrderBy(r => r.Ms))
    {
        Console.WriteLine($"  {r.Model,-22}  {r.Ms,6} ms   {r.Characters,6} chars");
    }
}

// Local helper for percentiles
long PercentileLocal(List<long> values, int percentile)
{
    if (values.Count == 0) return 0;
    var sorted = values.OrderBy(v => v).ToArray();
    var rank = (int)Math.Ceiling(percentile / 100.0 * sorted.Length);
    rank = Math.Clamp(rank, 1, sorted.Length);
    return sorted[rank - 1];
}

// Percentile summaries
Console.WriteLine("\n=== Percentiles ===");
foreach (var kvp in latencies.OrderBy(k => k.Key))
{
    var p50 = PercentileLocal(kvp.Value, 50);
    var p95 = PercentileLocal(kvp.Value, 95);
    var p99 = PercentileLocal(kvp.Value, 99);
    Console.WriteLine($"{kvp.Key}  p50={p50} ms  p95={p95} ms  p99={p99} ms");
}

// Simple record to hold results
record BenchResult(string Model, string Task, long Ms, int Characters);

// Singleton holder for FoundryLocalManager
static class FoundryManagerSingleton
{
    private static readonly SemaphoreSlim Gate = new(1, 1);
    private static FoundryLocalManager? _instance;

    public static async Task<FoundryLocalManager> GetAsync()
    {
        if (_instance != null) return _instance;

        await Gate.WaitAsync();
        try
        {
            if (_instance != null) return _instance;

            var config = new Configuration
            {
                AppName = "foundry-local-bench",
                LogLevel = Microsoft.AI.Foundry.Local.LogLevel.Information,
                Web = new Configuration.WebService
                {
                    Urls = Environment.GetEnvironmentVariable("FOUNDRY_LOCAL_ENDPOINT") ?? "http://127.0.0.1:55588"
                },
                AppDataDir = "./foundry_local_data",
                ModelCacheDir = "{AppDataDir}/model_cache",
                LogsDir = "{AppDataDir}/logs"
            };

            using var lf = LoggerFactory.Create(b => b.SetMinimumLevel(Microsoft.Extensions.Logging.LogLevel.Information));
            var logger = lf.CreateLogger("FoundrySingleton");

            await FoundryLocalManager.CreateAsync(config, logger);
            _instance = FoundryLocalManager.Instance;
        }
        finally
        {
            Gate.Release();
        }

        return _instance!;
    }
}
