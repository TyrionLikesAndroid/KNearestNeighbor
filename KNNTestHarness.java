import java.util.Random;

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
        data.splitValidationAndTestData(70,new Random(1223345466));
//        data.splitValidationAndTestData(70,null);

        System.out.println("\nStarting KNN analysis");

        // Create a data classifier with our data set and measure all of our test data.  This will calculate
        // the Euclidean distance for all of our test data points and save it in memory for classification.
        KNNClassifier classifier = new KNNClassifier(data);
        classifier.measureTestData();

        // Run the KNN tests with our different K values
        classifier.classifyTestSet(1);
        classifier.classifyTestSet(3);
        classifier.classifyTestSet(5);
        classifier.classifyTestSet(7);
        classifier.classifyTestSet(9);

        System.out.println("\nStarting CNN analysis");
        data.measureTrainingData();

        // Run the CNN tests with our different K values
        testCondensedNearestNeighbor(1, data, classifier);
        testCondensedNearestNeighbor(3, data, classifier);
        testCondensedNearestNeighbor(5, data, classifier);
        testCondensedNearestNeighbor(7, data, classifier);
        testCondensedNearestNeighbor(9, data, classifier);
    }

    static void testCondensedNearestNeighbor(int kValue, KNNDataSet data, KNNClassifier classifier)
    {
        System.out.println();
        data.resetCondensedTrainingData();

        boolean condenseComplete = false;
        while(! condenseComplete)
            condenseComplete = data.condenseTrainingData(kValue);
        data.confirmCondensedEquivalency(kValue);

        classifier.measureCondensedTestData(data.getCondensedTrainingData());
        classifier.classifyTestSet(kValue);
    }
}