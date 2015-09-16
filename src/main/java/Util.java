import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.io.IOException;

/**
 * Created by timothy.pratama on 16-Sep-15.
 */
public class Util {

    private static String pathDataSet = "dataSet/";

    public static Instances readARFF(String namaFile)
    {
        try
        {
            Instances dataSet;
            ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource(pathDataSet + namaFile);
            dataSet = dataSource.getDataSet();
            if(dataSet.classIndex() == -1)
            {
                dataSet.setClassIndex(dataSet.numAttributes()-1);
            }
            return dataSet;
        }

        catch (Exception e)
        {
            e.printStackTrace();
            return null;
        }
    }

    public static Instances readCSV(String namaFile)
    {
        try
        {
            CSVLoader csvLoader = new CSVLoader();
            csvLoader.setSource(new File(pathDataSet + namaFile));
            Instances dataSet = csvLoader.getDataSet();
            return dataSet;
        }

        catch (IOException e)
        {
            e.printStackTrace();
            return null;
        }
    }

    public static void main(String [] args)
    {
        System.out.println("Reading file from ARFF");
        Instances dataSet = Util.readARFF("weather.nominal.arff");
        System.out.println(dataSet.toString());

        System.out.println("\nReading file from CSV");
        dataSet = Util.readCSV("dataSet.csv");
        System.out.println(dataSet.toString());
    }
}
