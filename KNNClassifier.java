import java.util.*;

public class KNNClassifier {

    public static final float BENIGN = -1.0f;
    public static final float MALIGNANT = 1.0f;
    public static final int LABEL_INDEX = 30;
    KNNDataSet dataSet;
    HashMap<Integer, TreeSet<AbstractMap.SimpleEntry<Integer,Float>>> distanceCalcs;

    class KNNDistanceCompare implements Comparator<AbstractMap.SimpleEntry<Integer,Float>> {

        public int compare(AbstractMap.SimpleEntry<Integer,Float> a, AbstractMap.SimpleEntry<Integer,Float> b)
        {
            if(a.getValue() == b.getValue())
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
        // of our training data
        for(int i = 0; i < dataSet.getTestDataSize(); i++)
        {
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

        for(int i = 0; i < KNNDataSet.NUM_DATA_COLUMNS-1; i++)
            distanceSum += Math.pow(testRow.get(i)-trainingRow.get(i),2);

        return new AbstractMap.SimpleEntry<>(rowId, new Float(Math.sqrt(distanceSum)));
    }

    public void dumpNearestNeighbors(int testRowId)
    {
        Iterator<AbstractMap.SimpleEntry<Integer,Float>> iter = distanceCalcs.get(testRowId).iterator();
        while(iter.hasNext())
        {
            AbstractMap.SimpleEntry<Integer,Float> entry = iter.next();
            System.out.println("Neighbors for [" + testRowId + "] " + entry);
        }
    }

    public void classifyTestSet(int kValue)
    {
        System.out.println("Classifying Test Data for K[" + kValue + "]");

        int matchCount = 0;
        int mismatchCount = 0;
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

        System.out.println("\nTest result K[" + kValue + "] match=" + matchCount + " mismatch=" + mismatchCount);
        System.out.println("    truePositive=" + positiveMatchCount + " trueNegative=" + negativeMatchCount);
        System.out.println("    falsePositive=" + falsePositiveCount + " falseNegative=" + falseNegativeCount);
    }

    public float determineKNNLabel(int testRowId, int kValue)
    {
        int malignantCount = 0;
        int benignCount = 0;
        int count = 1;

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