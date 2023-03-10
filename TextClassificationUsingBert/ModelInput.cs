using Microsoft.ML.Data;

namespace TextClassificationUsingBert.Models;

public class ModelInput
{
    [LoadColumn(0)]
    [ColumnName(@"Content")]
    public string? Content { get; set; }

    [LoadColumn(1)]
    [ColumnName(@"Label")]
    public string? Label { get; set; }
}
