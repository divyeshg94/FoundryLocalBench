using System.Diagnostics;
using Microsoft.Extensions.Logging;
using System.Text.Json;
using System.Text;

// Config
int concurrency = 1; // adjust to run multiple requests in parallel later
string jsonlPath = "benchmarks.jsonl";
string csvPath = "benchmarks.csv";

// You can parameterize this later
var models = new[]
{
    "phi-4-mini",
    "phi-4-mini-reasoning",
    "qwen2.5-1.5b",
    "deepseek-r1-7b",
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
await File.WriteAllTextAsync(csvPath, "model,task,ms,chars,tokens,tokens_per_sec,cpu_pct,working_set_mb,response\n");

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

foreach (var (taskName, prompt) in tasks)
{
    Console.WriteLine();
    Console.WriteLine($"=== Task: {taskName} ===");

    foreach (var alias in models)
    {
        Console.WriteLine($"\nRunning {alias} …");

        var proc = Process.GetCurrentProcess();
        var cpuStart = proc.TotalProcessorTime;
        var sw = Stopwatch.StartNew();

        var responseText = $"Model {alias} responded to prompt of length {prompt.Length}.";

        sw.Stop();
        proc.Refresh();
        var cpuEnd = proc.TotalProcessorTime;

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
            response = content
        });
        await File.AppendAllTextAsync(jsonlPath, json + "\n");

        // CSV row (escape quotes/newlines for Excel)
        var responseCsv = content.Replace("\"", "\"\"").Replace("\r", " ").Replace("\n", " ");
        await File.AppendAllTextAsync(csvPath, $"{alias},{taskName},{ms},{chars},{tokens},{tokensPerSec:F2},{cpuPct:F2},{workingSetMb:F2},\"{responseCsv}\"\n");

        logger.LogInformation("{Alias} | {Task} | {Ms} ms | {Chars} chars | {Tokens} tok | {TokPerSec} tok/s | CPU {CpuPct}% | WS {WsMb} MB",
                              alias, taskName, ms, chars, tokens, tokensPerSec, cpuPct, workingSetMb);

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
