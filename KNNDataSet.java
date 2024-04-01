import java.io.BufferedReader;
import java.io.FileReader;

public class KNNDataSet {

    static final String DELIMETER = ",";
    static final int NUM_DATA_ROWS = 613;
    static final int NUM_DATA_COLUMNS = 31;
    final String dataFilePath;
    float [][] originalDataSet;
    float [][] trainingData;
    float [][] testData;

    public KNNDataSet(String dataFilePath)
    {
        this.dataFilePath = dataFilePath;
        originalDataSet = new float[NUM_DATA_COLUMNS][NUM_DATA_ROWS];
    }

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

            // This is the math loop where we calculate average, min, max for the column
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

            // This is the normalize loop where we normalize the data in place
            for (int i = 0; i <= NUM_DATA_ROWS-1; i++)
            {
                float value = originalDataSet[j][i];
                originalDataSet[j][i] = (value - average) / (max - min);
            }
        }
    }

    public void printDataSet()
    {
        for(int i = 0; i <= NUM_DATA_ROWS-1; i++)
        {
            System.out.println("Line " + i + ":");

            // This is the math loop where we calculate average, min, max for the column
            for (int j = 0; j <= NUM_DATA_COLUMNS-1; j++)
                System.out.print(originalDataSet[j][i] + ",");

            System.out.println("");
        }
    }
}
