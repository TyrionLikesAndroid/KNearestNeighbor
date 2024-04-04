import java.io.BufferedReader;
import java.io.FileReader;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Vector;

public class KNNDataSet {

    static final String DELIMETER = ",";    // Comma delimeter for the CSV file
    static final int NUM_DATA_ROWS = 613;   // Total number of CSV rows
    static final int NUM_DATA_COLUMNS = 31; // Total number of CSV columns
    final String dataFilePath;      // Path to our CSV filie
    float [][] originalDataSet;     // Data member for our original dataset
    float [][] trainingData;        // Data member for our training data
    float [][] testData;            // Data member for our test data

    public KNNDataSet(String dataFilePath)
    {
        this.dataFilePath = dataFilePath;
        originalDataSet = new float[NUM_DATA_COLUMNS][NUM_DATA_ROWS];
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