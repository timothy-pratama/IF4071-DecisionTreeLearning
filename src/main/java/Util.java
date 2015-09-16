import weka.core.Instances;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;

import java.io.File;
import java.io.IOException;

/**
 * Created by timothy.pratama on 16-Sep-15.
 */
public class Util {

    public static Instances readARFF(String path)
    {
        try
        {
            Instances dataSet;
            ConverterUtils.DataSource dataSource = new ConverterUtils.DataSource(path);
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

    public static Instances readCSV(String path)
    {
        try
        {
            CSVLoader csvLoader = new CSVLoader();
            csvLoader.setSource(new File(path));
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
        Instances dataSet = Util.readARFF("data/weather.nominal.arff");
        System.out.println(dataSet.toString());

        System.out.println("Reading file from CSV");
        dataSet = Util.readCSV("data/dataSet.csv");
        System.out.println(dataSet.toString());
    }
}
