using Microsoft.ML.Data;

namespace TextClassificationUsingBert.Models;

public class ModelIntermediary
{
    [ColumnName(@"Content")]
    public string? Content { get; set; }

    [ColumnName(@"WordsWithoutDefaultStopWords")]
    public string[]? WordsWithoutDefaultStopWords { get; set; }
}
