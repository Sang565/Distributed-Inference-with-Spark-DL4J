/*
Purpose: Run model inference with DL4J in local mode (local Spark)

*/

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import scala.Tuple2;

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class Model_Inference_localTesting {

    public static <E> void main(String[] args) throws Exception {
        // SANG: set log level
        Logger.getLogger("org.apache.spark").setLevel(Level.WARN);

        // 1. ---- Create a Spark Context/Session -----------------------
        SparkConf conf = new SparkConf();
        conf.setAppName("Model_Inference_localTesting");
        conf.setMaster("local[*]");

        JavaSparkContext sc = new JavaSparkContext(conf);

        // 2. ---- Load pre-trained Keras-based model -------------------
        System.out.println("---- load pre-trained model ----------");
        //String modelPath = "hdfs://afog-master:9000/part4-projects/model/model.h5";  //SANG: for HDFS storage
        //String localModelPath = "./src/main/resources/model/model.h5";
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("./src/main/resources/model/model.h5", true);

        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1)
                .batchSizePerWorker(32)
                .averagingFrequency(5).workerPrefetchNumBatches(2)
                .build();

        // 3. ---- Create a network on Spark (sparkNet) -------------------
        System.out.println("------ Create a sparkNet  ---------");
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, model, tm);

        // 4. ------ Read the test dataset (testData) --------------------------------
        System.out.println("------ Read the test dataset  ---------");
        //String datafilePath = "hdfs://afog-master:9000/part4-projects/datasets/test_data.csv"; //SANG: for HDFS storage
        String datafilePath = "./src/main/resources/datasets/test_data.csv"; //Origianlly exported fom Colab notebook - 26,486 records
        //String datafilePath = "./src/main/resources/datasets/test_data_small.csv";
        JavaRDD<String> rddString = sc.textFile(datafilePath);

        RecordReader recordReader = new CSVRecordReader(0,  ',');  //Read CSV files
        JavaRDD<List<Writable>> rddWritables = rddString.map(new StringToWritablesFunction(recordReader));

        // 5. ------ Create testData in JavaRDD<Dataset> --------------------------------
        System.out.println("------ Create testData in JavaRDD<Dataset>  ---------");
        //--- for regression loading data: multiple features columns --------------
        //Ref: https://deeplearning4j.konduit.ai/distributed-deep-learning/data-howto
        int firstColumnLabel = 0;    // First column index for label
        int lastColumnLabel = 0;    // Last column index for label
        JavaRDD<DataSet> testData = rddWritables.map(new DataVecDataSetFunction(firstColumnLabel, lastColumnLabel, true, null, null));

        // 6. ----- Make prediction --------------------
        System.out.println("------ Make prediction  ---------");
        //JavaPairRDD<String, INDArray> testPairs = testData.mapToPair(f-> new Tuple2("Occ_Predicted", f.getLabels() ));
        JavaPairRDD<String, INDArray> testPairs = testData.mapToPair(f-> new Tuple2("Predicted", f.getFeatures()));
        JavaPairRDD<String, INDArray> predictions = sparkNet.feedForwardWithKey(testPairs, 5);

        // --- output the predicted results ------
        List<DataSet> actualData = testData.collect(); //collect actual data
        AtomicInteger counter = new AtomicInteger(1);
        long startTime = System.nanoTime();

        predictions.collect().forEach(p->{
            //System.out.println(p._1 + ": " + p._2); //simple print
            //System.out.println(p._1 + ": " + p._2.getInt()); // Print Int values in Int only
            System.out.println("Row:" + counter + " ==> " + p._1 + ":" + p._2.getInt() + " - " + "Actual:" + actualData.get(counter.getAndIncrement()-1).getFeatures().getInt());
        });

        // ---------------------------------------------
        System.out.println("Total no. of records: " + predictions.count()); //20

        long endTime = System.nanoTime();
        long duration = (endTime - startTime);  //divide by 1,000,000 to get milliseconds
        System.out.println("Running time (ms): " + duration/1000000);

        System.out.println("********** DONE **********");

        // 7. save the predicted results in a file (in Hadoop text format part0000)
        //predictions.saveAsTextFile("./src/main/resources/inferences/");
    }
}
