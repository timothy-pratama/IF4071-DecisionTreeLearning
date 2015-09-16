import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.CSVLoader;
import weka.core.converters.ConverterUtils;
import weka.filters.Filter;
import weka.filters.supervised.instance.Resample;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;
import java.io.IOException;

/**
 * Created by timothy.pratama on 16-Sep-15.
 */
public class Util {

    private static String pathDataSet = "dataSet/";

    /**
     *
     * @param namaFile Nama file data set dengan format arff
     * @return Instances dari data set
     */
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

    /**
     * @param namaFile Nama file data set dengan format csv
     * @return Instances dari data set
     */
    public static Instances readCSV(String namaFile)
    {
        try
        {
            CSVLoader csvLoader = new CSVLoader();
            csvLoader.setSource(new File(pathDataSet + namaFile));
            Instances dataSet = csvLoader.getDataSet();
            if(dataSet.classIndex() == -1)
            {
                dataSet.setClassIndex(dataSet.numAttributes()-1);
            }
            return dataSet;
        }

        catch (IOException e)
        {
            e.printStackTrace();
            return null;
        }
    }

    /**
     * @param dataSet Data set yang akan di-filter
     * @param attributeIndex Atribut yang akan dibuang dari data set (bernilai dari 1 sampai jumlah atribut)
     * @return Data set dengan atribut ke attributeIndex sudah dibuang
     */
    public static Instances removeAttribute(Instances dataSet, int attributeIndex)
    {
        try
        {
            String options[] = new String[2];
            options[0] = "-R";
            options[1] = String.valueOf(attributeIndex);
            Remove remove = new Remove();
            remove.setOptions(options);
            remove.setInputFormat(dataSet);
            Instances newDataSet = Filter.useFilter(dataSet,remove);
            return newDataSet;
        }
        catch (Exception e)
        {
            e.printStackTrace();
            return null;
        }
    }

    public static Instances resampleDataSet(Instances dataSet)
    {
        try
        {
            Resample resample = new Resample();
            String filterOptions = "-B 0.0 -S 1 -Z 100.0";
            resample.setOptions(Utils.splitOptions(filterOptions));
            resample.setRandomSeed((int)System.currentTimeMillis());
            resample.setInputFormat(dataSet);
            Instances newDataSet = Filter.useFilter(dataSet,resample);
            return newDataSet;
        }
        catch (Exception e)
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
        System.out.println("Class Attribute: " + dataSet.attribute(dataSet.classIndex()));

        System.out.println("\nResampling data set");
        dataSet = Util.resampleDataSet(dataSet);
        System.out.println(dataSet.toString());

        System.out.println("\nReading file from CSV");
        dataSet = Util.readCSV("dataSet.csv");
        System.out.println(dataSet.toString());
        System.out.println("Class Attribute: " + dataSet.attribute(dataSet.classIndex()));

        System.out.println("\nRemoving class attributes");
        dataSet = readARFF("weather.nominal.arff");
        dataSet = Util.removeAttribute(dataSet,dataSet.numAttributes());
        System.out.println(dataSet.toString());
    }
}
