using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Serilog;
using Serilog.Sinks.SystemConsole.Themes;
using System.Data;
using TextClassificationUsingBert.Models;

Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Debug()
    .MinimumLevel.Override("Microsoft", Serilog.Events.LogEventLevel.Warning)
    .MinimumLevel.Override("System", Serilog.Events.LogEventLevel.Warning)
    .WriteTo.Console(outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}", theme: SystemConsoleTheme.Colored)
    .CreateLogger();

const string ModelName = "subjects.zip";

MLContext mlContext = new()
{
    GpuDeviceId = 0,
    FallbackToCpu = true
};

mlContext.Log += (_, e) =>
{
    if (e.Source.StartsWith("TextClassificationTrainer") || e.Source.StartsWith("NasBertTrainer"))
    {
        Log.Debug(e.Message);
    }
};

// Load the dataset
Log.Information("Loading data...");
IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
    //"subjects-questions-10k.tsv",
    "subjects-questions.tsv",
    hasHeader: true,
    separatorChar: '\t'
);
DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 2023);
IDataView trainData = dataSplit.TrainSet;
IDataView testData = dataSplit.TestSet;

////var textPipeline = mlContext.Transforms.Text.TokenizeIntoWords("Text", "Content")
////    .Append(mlContext.Transforms.Text.RemoveStopWords("WordsWithoutDefaultStopWords", "Text", stopwords: new[] { "{", "}", "(", ")", ".", ",", "A", "B", "C", "D", "A.", "B.", "C.", "D.", "A)", "B)", "C)", "D)" }))
////    .Append(mlContext.Transforms.Text.TokenizeIntoWords("WordsWithoutDefaultStopWords"));

////var emptySamples = new List<ModelInput>();
////var emptyDataView = mlContext.Data.LoadFromEnumerable(emptySamples);

////var textTransformer = textPipeline.Fit(emptyDataView);

////var predictions = textTransformer.Transform(trainData);
////var outputData = mlContext.Data.CreateEnumerable<ModelIntermediary>(predictions, false);
////var processedData = outputData.Select(d => string.Join(" ", d.WordsWithoutDefaultStopWords));

// Create a pipeline for training the model
var pipeline = mlContext.Transforms.Conversion.MapValueToKey(
        outputColumnName: @"Label",
        inputColumnName: @"Label")
    .Append(mlContext.MulticlassClassification.Trainers.TextClassification(
        labelColumnName: @"Label", 
        sentence1ColumnName: @"Content"))
    .Append(mlContext.Transforms.Conversion.MapKeyToValue(
        outputColumnName: @"PredictedLabel", 
        inputColumnName: @"PredictedLabel"));

Log.Information("Training model...");
////ITransformer model = pipeline.Fit(trainData);
////var processedTrainData = mlContext.Data.LoadFromEnumerable(processedData);
Log.Information("Model training is complete.");

//ITransformer model = pipeline.Fit(trainData);
//mlContext.Model.Save(model, trainData.Schema, ModelName);
//Log.Information($"Model {ModelName} is saved.");
var model = mlContext.Model.Load(ModelName, out var _);

////Log.Information("Evaluating model performance...");

////// We need to apply the same transformations to our test set so it can be evaluated via the resulting model
////IDataView transformedTest = model.Transform(testData);
////MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(transformedTest);

////// Display Metrics
////Log.Information($"Macro Accuracy: {metrics.MacroAccuracy}");
////Log.Information($"Micro Accuracy: {metrics.MicroAccuracy}");
////Log.Information($"Log Loss: {metrics.LogLoss}");
////Log.Information(metrics.ConfusionMatrix.GetFormattedConfusionTable());

// Generate a prediction engine
PredictionEngine<ModelInput, ModelOutput> engine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

Console.WriteLine("Evaluation is complete.");

string content;
do
{
    Console.WriteLine();
    Console.WriteLine("Give me some content to predict... (type 'Q' or 'q' to exit)");
    content = Console.ReadLine()!;

    ModelInput sampleData = new() { Content = content };
    ModelOutput result = engine.Predict(sampleData);

    Console.WriteLine($"Matched {result.PredictedLabel} with score of {result.Score!.Select(Math.Abs).Max()}");
    Console.WriteLine();
}
while (!string.IsNullOrWhiteSpace(content) && content.Equals("Q", StringComparison.InvariantCultureIgnoreCase));
