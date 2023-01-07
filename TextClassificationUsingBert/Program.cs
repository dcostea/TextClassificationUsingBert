using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.TorchSharp;
using Serilog;
using Serilog.Sinks.SystemConsole.Themes;
using System.Data;


Log.Logger = new LoggerConfiguration()
    .MinimumLevel.Debug()
    .MinimumLevel.Override("Microsoft", Serilog.Events.LogEventLevel.Warning)
    .MinimumLevel.Override("System", Serilog.Events.LogEventLevel.Warning)
    .WriteTo.Console(outputTemplate: "[{Timestamp:HH:mm:ss} {Level:u3}] {Message:lj}{NewLine}", theme: SystemConsoleTheme.Colored)
    .CreateLogger();

// Initialize MLContext
MLContext mlContext = new()
{
    GpuDeviceId = 0,
    FallbackToCpu = true
};

mlContext.Log += (_, e) =>
{
    if (e.Source.StartsWith("TextClassificationTrainer"))
    {
        Log.Debug(e.Message);
    }
};

// Load the data source
Log.Information("Loading data...");
IDataView dataView = mlContext.Data.LoadFromTextFile<ModelInput>(
    //"subjects-questions-10k.tsv",
    "subjects-questions.tsv",
    hasHeader: true,
    separatorChar: '\t'
);

/** MODEL TRAINING ****************************************************************************/

// To evaluate the effectiveness of machine learning models we split them into a training set for fitting
// and a testing set to evaluate that trained model against unknown data
DataOperationsCatalog.TrainTestData dataSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2, seed: 1234);
IDataView trainData = dataSplit.TrainSet;
IDataView testData = dataSplit.TestSet;

////var textPipeline = mlContext.Transforms.Text.TokenizeIntoWords("Text", "Content")
////    .Append(mlContext.Transforms.Text.RemoveStopWords("WordsWithoutDefaultStopWords", "Text", stopwords: new[] { "{", "}", "(", ")", ".", ",", "A", "B", "C", "D", "A.", "B.", "C.", "D.", "A)", "B)", "C)", "D)" }))
////    .Append(mlContext.Transforms.Text.TokenizeIntoWords("WordsWithoutDefaultStopWords"));

////var emptySamples = new List<ModelInput>();
////// Convert sample list to an empty IDataView.
////var emptyDataView = mlContext.Data.LoadFromEnumerable(emptySamples);

////// Fit to data.
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

// Train the model using the pipeline
Log.Information("Training model...");
////ITransformer model = pipeline.Fit(trainData);
////var processedTrainData = mlContext.Data.LoadFromEnumerable(processedData);

ITransformer model = pipeline.Fit(trainData);
mlContext.Model.Save(model, trainData.Schema, "subjects.zip");
////var model = mlContext.Model.Load("subjects.zip", out var _);
//new string[] { "A)", "B)", "C)", "D)" }
/** MODEL EVALUATION **************************************************************************/

// Evaluate the model's performance against the TEST data set
Log.Information("Evaluating model performance...");

// We need to apply the same transformations to our test set so it can be evaluated via the resulting model
IDataView transformedTest = model.Transform(testData);
MulticlassClassificationMetrics metrics = mlContext.MulticlassClassification.Evaluate(transformedTest);

// Display Metrics
Log.Information($"Macro Accuracy: {metrics.MacroAccuracy}");
Log.Information($"Micro Accuracy: {metrics.MicroAccuracy}");
Log.Information($"Log Loss: {metrics.LogLoss}");
Log.Information(metrics.ConfusionMatrix.GetFormattedConfusionTable());

/** PREDICTION GENERATION *********************************************************************/

// Generate a prediction engine
Log.Information("Creating prediction engine...");
PredictionEngine<ModelInput, ModelOutput> engine = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

Console.WriteLine("Ready to generate predictions.");

// Generate a series of predictions based on user input
string input;
do
{
    Console.WriteLine();
    Console.WriteLine("What subjet do you have? (Type Q to Quit)");
    input = Console.ReadLine()!;

    // Get a prediction
    ModelInput sampleData = new(input);
    ModelOutput result = engine.Predict(sampleData);

    // Print classification
    //float maxScore = result.Score[(uint)result.PredictedLabel];
    Console.WriteLine($"Matched intent {result.PredictedLabel} with score of {result.Score.Select(s => Math.Abs(s)).Max()}");
    Console.WriteLine();
}
while (!string.IsNullOrWhiteSpace(input) && input.ToLowerInvariant() != "q");
