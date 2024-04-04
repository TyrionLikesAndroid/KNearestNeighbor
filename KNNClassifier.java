import java.text.DecimalFormat;
import java.util.*;

public class KNNClassifier {

    public static final float BENIGN = -1.0f;   // constant for benign value
    public static final float MALIGNANT = 1.0f; // constant for malignant value
    public static final int LABEL_INDEX = 30;   // constant for the position of the label in the CSV file
    KNNDataSet dataSet; // Our dataset
    HashMap<Integer, TreeSet<AbstractMap.SimpleEntry<Integer,Float>>> distanceCalcs;    // Our distance measurements

    // Comparator for the tree so that we will sort the lowest cost heuristics to the top
    static class KNNDistanceCompare implements Comparator<AbstractMap.SimpleEntry<Integer,Float>> {

        public int compare(AbstractMap.SimpleEntry<Integer,Float> a, AbstractMap.SimpleEntry<Integer,Float> b)
        {
            if(Objects.equals(a.getValue(), b.getValue()))
                return 0;

            // Sort the lowest cost heuristics to the top
            return (a.getValue() > b.getValue()) ? 1 : -1;
        }
    }

    public KNNClassifier(KNNDataSet data)
    {
        this.dataSet = data;
        this.distanceCalcs = new HashMap<>();
    }

    public void measureTestData()
    {
        // Loop through the test data and make Euclidean distance measurements against all
        // of our training data.  We will do all the distance math one time, then use it repeatedly for different
        // values of K in the test harness.
        for(int i = 0; i < dataSet.getTestDataSize(); i++)
        {
            // Grab a row of test data and create a tree set where we will hold the calculated distances from this
            // test row to every training row.
            Vector<Float> testRow = dataSet.getTestDataRow(i);
            TreeSet<AbstractMap.SimpleEntry<Integer,Float>> singleRowResults = new TreeSet<>(new KNNDistanceCompare());

            // Measure the distance between this test row and every row in the training set
            for(int j = 0; j < dataSet.getTrainingDataSize(); j++)
            {
                Vector<Float> trainingRow = dataSet.getTrainingDataRow(j);
                AbstractMap.SimpleEntry<Integer,Float> distance = calculateDistance(j, testRow, trainingRow);
                singleRowResults.add(distance);
            }

            // Assign the priority queue to the class level dictionary
            distanceCalcs.put(i,singleRowResults);
        }
    }

    private AbstractMap.SimpleEntry<Integer,Float> calculateDistance(int rowId, Vector<Float>testRow, Vector<Float> trainingRow)
    {
        float distanceSum = 0.0f;

        // This is our distance function.  Square and sum the delta from each column, then get the total square root below
        for(int i = 0; i < KNNDataSet.NUM_DATA_COLUMNS-1; i++)
            distanceSum += Math.pow(testRow.get(i)-trainingRow.get(i),2);

        return new AbstractMap.SimpleEntry<>(rowId, new Float(Math.sqrt(distanceSum)));
    }

    public void dumpNearestNeighbors(int testRowId)
    {
        // A helper function to confirm that our tree is ordering correctly based on the heuristic
        Iterator<AbstractMap.SimpleEntry<Integer,Float>> iter = distanceCalcs.get(testRowId).iterator();
        while(iter.hasNext())
        {
            AbstractMap.SimpleEntry<Integer,Float> entry = iter.next();
            System.out.println("Neighbors for [" + testRowId + "] " + entry);
        }
    }

    public void classifyTestSet(int kValue)
    {
        // This is our main classification function.  We iterate through the test data, get the test label and
        // the calculated label, then collect our metrics for the confusion matrix and accuracy.

        System.out.println("\nClassifying Test Data for K[" + kValue + "]");

        float matchCount = 0;
        float mismatchCount = 0;
        int negativeMatchCount = 0;
        int positiveMatchCount = 0;
        int falsePositiveCount = 0;
        int falseNegativeCount = 0;

        for(int i = 0; i < dataSet.getTestDataSize(); i++)
        {
            // Determine the correct label for this test row
            float label = dataSet.getTestDataRow(i).get(LABEL_INDEX);

            // Calculate the KNN label for this test row
            float knnLabel = determineKNNLabel(i, kValue);

            // Compare our calculated label with our test label.  Negative in this case is BENIGN and
            // positive is MALIGNANT
            if(label == knnLabel)
            {
                matchCount++;
                if(label == BENIGN)
                    negativeMatchCount++;
                else
                    positiveMatchCount++;
            }
            else
            {
                mismatchCount++;
                if(label == BENIGN)
                    falsePositiveCount++;
                else
                    falseNegativeCount++;
            }
        }

        DecimalFormat numberFormat = new DecimalFormat("#.00");
        System.out.println("Test result K[" + kValue + "] match=" + matchCount + " mismatch=" + mismatchCount +
            " accuracy=" + numberFormat.format(matchCount/(matchCount + mismatchCount)*100));
        System.out.println("    truePositive=" + positiveMatchCount + " falsePositive=" + falsePositiveCount);
        System.out.println("    falseNegative=" + falseNegativeCount + " trueNegative=" + negativeMatchCount);
    }

    public float determineKNNLabel(int testRowId, int kValue)
    {
        int malignantCount = 0;
        int benignCount = 0;
        int count = 1;

        // Simply count the number of malignant neighbors versus the number of benign neighbors for our
        // K total of nearest neighbors
        Iterator<AbstractMap.SimpleEntry<Integer,Float>> iter = distanceCalcs.get(testRowId).iterator();
        while(iter.hasNext() && (count <= kValue))
        {
            AbstractMap.SimpleEntry<Integer,Float> entry = iter.next();
            count++;

            // Count the labels for the training entries
            float trainingLabel = dataSet.getTrainingDataRow(entry.getKey()).get(LABEL_INDEX);
            if(trainingLabel == MALIGNANT)
                malignantCount++;
            else if(trainingLabel == BENIGN)
                benignCount++;
            else
                System.out.println("ERROR: Label with unexpected value");
        }

        if(malignantCount > benignCount)
            return MALIGNANT;
        else
            return BENIGN;
    }
}