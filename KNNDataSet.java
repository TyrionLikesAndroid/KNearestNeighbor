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

    LinkedList<Integer> condensedTrainingData;  // Data member for training data

    public KNNDataSet(String dataFilePath)
    {
        this.dataFilePath = dataFilePath;
        originalDataSet = new float[NUM_DATA_COLUMNS][NUM_DATA_ROWS];

        condensedTrainingData = new LinkedList<>();
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

    public boolean condenseTrainingData(int kValue)
    {
        // Initialize our training data size and condensed data list if this is the first iteration
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
            Vector<Float> focalRow = getTrainingDataRow(focalRowId);

            // Determine the KNN label for the focal row versus the full training set
            Iterator<Integer> fullTrainingIter = trainingIndex.iterator();
            float closest = 9999.0f;
            int closestTrainingIndex = 0;
            float fullTrainingClassifier = KNNClassifier.BENIGN;
            while(fullTrainingIter.hasNext())
            {
                // Don't compare a row against itself
                int fullTrainingRowId = fullTrainingIter.next();
                if(fullTrainingRowId == focalRowId)
                    continue;

                // Get the training row for this iteration
                Vector<Float> fullTrainingRow = getTrainingDataRow(fullTrainingRowId);

                // This is our distance function.  Square and sum the delta from each column
                float testDistance = 0.0f;
                for(int i = 0; i < KNNDataSet.NUM_DATA_COLUMNS-1; i++)
                    testDistance += Math.pow(fullTrainingRow.get(i) - focalRow.get(i),2);

                // Determine if this is our closest neighbor, if so get the label from it
                if(testDistance < closest)
                {
                    fullTrainingClassifier = fullTrainingRow.get(KNNClassifier.LABEL_INDEX);
                    closest = testDistance;
                    closestTrainingIndex = fullTrainingRowId;
                    //System.out.println("Row [" + focalRowId + "] closest distance[" + closest + "] Full Training Label=" + fullTrainingClassifier);
                }
            }

            // Determine the KNN label for the focal row versus the condensed training set
            // Iterate through our index representation of the remaining training set
            closest = 9999.0f;
            float condensedTrainingClassifier = KNNClassifier.BENIGN;
            Iterator<Integer> iterCondensed = condensedTrainingData.iterator();
            while(iterCondensed.hasNext())
            {
                // Don't compare a row against itself
                int condensedRowId = iterCondensed.next();
                if(condensedRowId == focalRowId)
                    continue;

                // Get the condensed row for this iteration
                Vector<Float> condensedRow = getTrainingDataRow(condensedRowId);

                // This is our distance function.  Square and sum the delta from each column
                float testDistance = 0.0f;
                for(int i = 0; i < KNNDataSet.NUM_DATA_COLUMNS-1; i++)
                    testDistance += Math.pow(condensedRow.get(i) - focalRow.get(i),2);

                // Determine if this is our closest neighbor, if so get the label from it
                if(testDistance < closest)
                {
                    condensedTrainingClassifier = condensedRow.get(KNNClassifier.LABEL_INDEX);
                    closest = testDistance;
                    //System.out.println("Row [" + focalRowId + "] closest distance[" + closest + "] Condensed Training Label=" + condensedTrainingClassifier);
                }
            }

            // Determine if the condensed label matches the training label
            if(condensedTrainingClassifier != fullTrainingClassifier)
            {
                // We need to add this point to our condensed test set
                condensedTrainingData.add(closestTrainingIndex);
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
            Vector<Float> focalRow = getTrainingDataRow(focalRowId);

            // Determine the KNN label for the focal row versus the full training set
            Iterator<Integer> iter2 = trainingIndex.iterator();
            float closest = 9999.0f;
            float fullTrainingClassifier = KNNClassifier.BENIGN;
            while(iter2.hasNext())
            {
                // Don't compare a row against itself
                int testRowId = iter2.next();
                if(testRowId == focalRowId)
                    continue;

                // Get the training row for this iteration
                Vector<Float> testRow = getTrainingDataRow(testRowId);

                // This is our distance function.  Square and sum the delta from each column
                float testDistance = 0.0f;
                for(int i = 0; i < KNNDataSet.NUM_DATA_COLUMNS-1; i++)
                    testDistance += Math.pow(testRow.get(i) - focalRow.get(i),2);

                // Determine if this is our closest neighbor, if so get the label from it
                if(testDistance < closest)
                {
                    fullTrainingClassifier = testRow.get(KNNClassifier.LABEL_INDEX);
                    closest = testDistance;
                    //System.out.println("Row [" + trainingRowId + "] closest distance[" + closest + "] Full Training Label=" + fullTrainingClassifier);
                }
            }

            // Determine the KNN label for the focal row versus the condensed training set
            // Iterate through our index representation of the remaining training set
            closest = 9999.0f;
            float condensedTrainingClassifier = KNNClassifier.BENIGN;
            Iterator<Integer> iterCondensed = condensedTrainingData.iterator();
            while(iterCondensed.hasNext())
            {
                // Don't compare a row against itself
                int condensedRowId = iterCondensed.next();
                if(condensedRowId == focalRowId)
                    continue;

                Vector<Float> condensedRow = getTrainingDataRow(condensedRowId);

                // This is our distance function.  Square and sum the delta from each column
                float testDistance = 0.0f;
                for(int i = 0; i < KNNDataSet.NUM_DATA_COLUMNS-1; i++)
                    testDistance += Math.pow(condensedRow.get(i) - focalRow.get(i),2);

                // Determine if this is our closest neighbor, if so get the label from it
                if(testDistance < closest)
                {
                    condensedTrainingClassifier = condensedRow.get(KNNClassifier.LABEL_INDEX);
                    closest = testDistance;
                    //System.out.println("Row [" + trainingRowId + "] closest distance[" + closest + "] Condensed Training Label=" + condensedTrainingClassifier);
                }
            }

            // Determine if the condensed label matches the training label
            if(condensedTrainingClassifier != fullTrainingClassifier)
            {
                System.out.println("MISMATCH: row [" + focalRow + "] is not training equivalent");
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