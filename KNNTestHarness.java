public class KNNTestHarness {

    public static void main(String[] args)
    {
        KNNDataSet data = new KNNDataSet("data/wdbc.data.mb.csv");

        // Load the data into memory from the CSV file
        if(data.load())
            System.out.println("Data file loaded successfully");

        // Print the data before we normalize it
//        System.out.println("\nORIGINAL DATA:");
//        data.printDataSet();

        // Normalize the data
        data.normalize();

        // Print the data after we normalize it
//        System.out.println("\nNORMALIZED DATA:");
//        data.printDataSet();

        // Split up the training and test data
        data.splitValidationAndTestData(70,false);

        // Create a data classifier with our data set and measure all of our test data.  This will calculate
        // the Euclidean distance for all of our test data points and save it in memory for classification.
        KNNClassifier classifier = new KNNClassifier(data);
        classifier.measureTestData();

        // Print our classifier data for test purposes
//        classifier.dumpNearestNeighbors(0);
//        classifier.dumpNearestNeighbors(1);

        // Classify the test data with our different K values
        classifier.classifyTestSet(1);
        classifier.classifyTestSet(3);
        classifier.classifyTestSet(5);
        classifier.classifyTestSet(7);
        classifier.classifyTestSet(9);
    }
}