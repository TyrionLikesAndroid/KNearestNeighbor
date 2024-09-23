import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

public class KNNDataSet {

    static final String DELIMETER = ",";    // Comma delimeter for the CSV file
    static final int NUM_DATA_ROWS = 613;   // Total number of CSV rows
    static final int NUM_DATA_COLUMNS = 31; // Total number of CSV columns
    final String dataFilePath;      // Path to our CSV filie
    float [][] originalDataSet;     // Data member for our original dataset
    float [][] trainingData;        // Data member for our training data
    float [][] testData;            // Data member for our test data
    HashMap<Integer, TreeSet<AbstractMap.SimpleEntry<Integer,Float>>> distanceCalcs;    // Training distances
    HashMap<Integer, TreeSet<AbstractMap.SimpleEntry<Integer,Float>>> condensedDistanceCalcs;    // Condensed distances

    LinkedList<Integer> condensedTrainingData;  // Data member for training data

    public KNNDataSet(String dataFilePath)
    {
        this.dataFilePath = dataFilePath;
        this.originalDataSet = new float[NUM_DATA_COLUMNS][NUM_DATA_ROWS];

        this.condensedTrainingData = new LinkedList<>();
        this.distanceCalcs = new HashMap<>();
        this.condensedDistanceCalcs = new HashMap<>();
    }

    // Function to load our CSV file and parse the lines by the comma delimiter.  We take the parsed value
    // and stuff it into our data set attribute
    public boolean load()
    {
        boolean out = true;

        try
        {
            int i = 0;
            String line;
            BufferedReader br = new BufferedReader(new FileReader(dataFilePath));

            while ((line = br.readLine()) != null)
            {
                String[] values = line.split(DELIMETER);

                for(int j = 0; j <= NUM_DATA_COLUMNS-1; j++)
                    originalDataSet[j][i] = Float.parseFloat(values[j]);

                i++;
            }

            br.close();
        }
        catch(Exception e)
        {
            e.printStackTrace();
            out = false;
        }

        return out;
    }

    public void normalize()
    {
        // Normalize the code in place, there is really no reason to keep the original data.
        // Loop through each column and find the mean/min/max, then use the values to normalize
        // each row in that column
        for(int j = 0; j <= NUM_DATA_COLUMNS-2; j++)  // Don't normalize the label column
        {
            float average = 0.0f;
            float sum = 0.0f;
            float min = 99999999.0f;
            float max = 0.0f;

            // This is the math loop where we calculate average, min, max for the focal column
            for (int i = 0; i <= NUM_DATA_ROWS-1; i++) {
                float value = originalDataSet[j][i];
                sum += value;
                if (value > max)
                    max = value;
                if (value < min)
                    min = value;
            }

            average = sum / NUM_DATA_ROWS;
            //System.out.println("Column " + j + ": sum[" + sum + "] average[" + average + "] + " +
            //        "min[" + min + "] max[" + max + "]");

            // This is the normalize loop where we normalize the data in place for the focal column
            for (int i = 0; i <= NUM_DATA_ROWS-1; i++)
            {
                float value = originalDataSet[j][i];
                originalDataSet[j][i] = (value - average) / (max - min);
            }
        }
    }

    public void resetCondensedTrainingData()
    {
        condensedTrainingData.clear();
    }

    public void measureTrainingData()
    {
        // Loop through the training data and make Euclidean distance measurements against all the other
        // data points.  We will do all the distance math one time, then use it repeatedly for different
        // values of K in the test harness.
        for(int i = 0; i < getTrainingDataSize(); i++)
        {
            // Grab a row of training data and create a tree set where we will hold the calculated distances from this
            // training row to every other training row.
            Vector<Float> testRow = getTrainingDataRow(i);
            TreeSet<AbstractMap.SimpleEntry<Integer,Float>> singleRowResults = new TreeSet<>(new KNNClassifier.KNNDistanceCompare());

            // Measure the distance between this test row and every row in the training set
            for(int j = 0; j < getTrainingDataSize(); j++)
            {
                if(i == j) continue;

                Vector<Float> trainingRow = getTrainingDataRow(j);
                AbstractMap.SimpleEntry<Integer,Float> distance = KNNClassifier.calculateDistance(j, testRow, trainingRow);
                singleRowResults.add(distance);
            }

            // Assign the priority queue to the class level dictionary
            distanceCalcs.put(i,singleRowResults);
        }
    }

    public void measureCondensedTrainingData()
    {
        for(int i = 0; i < getTrainingDataSize(); i++)
        {
            // Grab a row of training data and create a tree set where we will hold the calculated distances from this
            // training row to every other training row.
            Vector<Float> testRow = getTrainingDataRow(i);
            TreeSet<AbstractMap.SimpleEntry<Integer, Float>> singleRowResults = new TreeSet<>(new KNNClassifier.KNNDistanceCompare());

            Iterator<Integer> condensedIter = condensedTrainingData.iterator();
            while(condensedIter.hasNext())
            {
                int focalRowId = condensedIter.next();
                if(focalRowId == i) continue;

                Vector<Float> trainingRow = getTrainingDataRow(focalRowId);
                AbstractMap.SimpleEntry<Integer,Float> distance = KNNClassifier.calculateDistance(focalRowId, testRow, trainingRow);
                singleRowResults.add(distance);
            }

            // Assign the priority queue to the class level dictionary
            condensedDistanceCalcs.put(i,singleRowResults);
        }
    }

    public boolean condenseTrainingData(int kValue)
    {
        // Initialize our condensed data size and list if this is the first iteration
        int condensedDataSize = condensedTrainingData.size();
        if(condensedDataSize == 0)
        {
            int randomRow = 0;
            for (int x = 0; x < kValue; x++)
            {
                randomRow = (int) (Math.random() * getTrainingDataSize());
                condensedTrainingData.add(randomRow);
                System.out.println("Seeding condensed list with [" + randomRow + "]");
            }

            measureCondensedTrainingData();
        }

        // Build a local training data index to work with
        LinkedList<Integer> trainingIndex = new LinkedList<>();
        for(int x = 0; x < getTrainingDataSize(); x++)
            trainingIndex.add(x);

        // Iterate through our index representation of the remaining training set
        Iterator<Integer> trainingIter = trainingIndex.iterator();
        while(trainingIter.hasNext())
        {
            int focalRowId = trainingIter.next();

            // Calculate the KNN label for this training row relative to its proximity to other training data
            float fullTrainingClassifier = KNNClassifier.determineKNNLabel(focalRowId, kValue, distanceCalcs, this);
            int closestTrainingIndex = distanceCalcs.get(focalRowId).first().getKey();

            // Determine the KNN label for this training row versus the condensed training set
            // Iterate through our index representation of the remaining training set
            float condensedTrainingClassifier = KNNClassifier.BENIGN;
            Iterator<Integer> iterCondensed = condensedTrainingData.iterator();
            while(iterCondensed.hasNext())
            {
                // Don't compare a row against itself
                int condensedRowId = iterCondensed.next();
                if(condensedRowId == focalRowId)
                    continue;

                condensedTrainingClassifier = KNNClassifier.determineKNNLabel(focalRowId, kValue, condensedDistanceCalcs, this);
            }

            // Determine if the condensed label matches the training label
            if(condensedTrainingClassifier != fullTrainingClassifier)
            {
                // We need to add this point to our condensed test set
                condensedTrainingData.add(closestTrainingIndex);
                measureCondensedTrainingData();
                System.out.println("MISMATCH: Add row [" + closestTrainingIndex + "] to condensed list to fix row [" + focalRowId + "]");
            }
        }

        System.out.println("Condensed training data size[" + condensedTrainingData.size() + "]");

        return (condensedDataSize == condensedTrainingData.size());
    }

    public void confirmCondensedEquivalency(int kValue)
    {
        System.out.println("Confirming training equivalency with condense set size [" + condensedTrainingData.size() + "]");

        // Build a local training data index to work with
        LinkedList<Integer> trainingIndex = new LinkedList<>();
        for(int x = 0; x < getTrainingDataSize(); x++)
            trainingIndex.add(x);

        // Iterate through our index representation of the remaining training set
        Iterator<Integer> trainingIter = trainingIndex.iterator();
        while(trainingIter.hasNext())
        {
            int focalRowId = trainingIter.next();

            // Calculate the KNN label for this training row relative to its proximity to other training data
            float fullTrainingClassifier = KNNClassifier.determineKNNLabel(focalRowId, kValue, distanceCalcs, this);
            int closestTrainingIndex = distanceCalcs.get(focalRowId).first().getKey();

            // Determine the KNN label for this training row versus the condensed training set
            // Iterate through our index representation of the remaining training set
            float condensedTrainingClassifier = KNNClassifier.BENIGN;
            Iterator<Integer> iterCondensed = condensedTrainingData.iterator();
            while(iterCondensed.hasNext())
            {
                // Don't compare a row against itself
                int condensedRowId = iterCondensed.next();
                if(condensedRowId == focalRowId)
                    continue;

                condensedTrainingClassifier = KNNClassifier.determineKNNLabel(focalRowId, kValue, condensedDistanceCalcs, this);
            }

            // Determine if the condensed label matches the training label
            if(condensedTrainingClassifier != fullTrainingClassifier)
            {
                System.out.println("MISMATCH: row [" + focalRowId + "] is not training equivalent");
                return;
            }
        }

        System.out.println("Condensed points are training equivalent");
    }

    public void printDataSet()
    {
        // Helper function to visualize the data that has been loaded into our data set member
        for(int i = 0; i <= NUM_DATA_ROWS-1; i++)
        {
            System.out.println("Line " + i + ":");

            // This is the math loop where we calculate average, min, max for the column
            for (int j = 0; j <= NUM_DATA_COLUMNS-1; j++)
                System.out.print(originalDataSet[j][i] + ",");

            System.out.println();
        }
    }

    public void splitValidationAndTestData(int trainingPercent, boolean randomFlag)
    {
        // Determine how many rows go into the training set and the test set
        int trainingDataSize = NUM_DATA_ROWS * trainingPercent/100;
        int testDataSize = NUM_DATA_ROWS - trainingDataSize;

        System.out.println("Splitting data into training [" + trainingDataSize + "] and test[" +
                testDataSize + "]");

        // Create the arrays in memory for our split datasets
        trainingData = new float[NUM_DATA_COLUMNS][trainingDataSize];
        testData = new float[NUM_DATA_COLUMNS][testDataSize];

        // Create a linked list with all the data keys and shuffle if randomFlag is true
        LinkedList<Integer> indexList = new LinkedList<>();
        for(int i = 0; i <= NUM_DATA_ROWS-1; i++)
            indexList.add(i);
        if(randomFlag)
            Collections.shuffle(indexList);

        // Create an iterator for the indexList
        Iterator<Integer> indexIter = indexList.iterator();

        // Copy the data from the original data into the new data sets
        for(int i = 0; i <= NUM_DATA_ROWS-1; i++)
        {
            int originalRow = indexIter.next();
            for(int k = 0; k <= NUM_DATA_COLUMNS-1; k++)
            {
                if(i < trainingDataSize)
                    trainingData[k][i] = originalDataSet[k][originalRow];
                else
                    testData[k][i-trainingDataSize] = originalDataSet[k][originalRow];
            }
        }
    }

    public int getTrainingDataSize()
    {
        return trainingData[0].length;
    }

    public int getTestDataSize()
    {
        return testData[0].length;
    }

    public LinkedList<Integer> getCondensedTrainingData()
    {
        return condensedTrainingData;
    }

    public Vector<Float> getTrainingDataRow(int rowId)
    {
        Vector<Float> out = new Vector<>();

        for(int k = 0; k <= NUM_DATA_COLUMNS-1; k++)
            out.add(k, trainingData[k][rowId]);

        return out;
    }

    public Vector<Float> getTestDataRow(int rowId)
    {
        Vector<Float> out = new Vector<>();

        for(int k = 0; k <= NUM_DATA_COLUMNS-1; k++)
            out.add(k, testData[k][rowId]);

        return out;
    }
}