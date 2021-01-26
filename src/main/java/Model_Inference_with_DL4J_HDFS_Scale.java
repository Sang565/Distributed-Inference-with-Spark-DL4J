/*
Purpose: Run model inference with DL4J for distributed mode on AFog Spark cluster
         --> used for scalability testing --> not display final predicted results in terminal

*/

import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
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
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.spark.api.TrainingMaster;
import org.deeplearning4j.spark.datavec.DataVecDataSetFunction;
import org.deeplearning4j.spark.impl.multilayer.SparkDl4jMultiLayer;
import org.deeplearning4j.spark.impl.paramavg.ParameterAveragingTrainingMaster;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import scala.Tuple2;

import java.io.BufferedInputStream;
import java.net.URI;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;

public class Model_Inference_with_DL4J_HDFS_Scale {

    public static <E> void main(String[] args) throws Exception {
        //--- Do comment these lines below if run with the local mode
        if (args.length != 3) {
            System.out.println("Usage:  Model_Inference_with_DL4J  \"spark://afog-master:7077\", <model-path> e.g. \"/model\"  , <dataset-path> e.g. \"/dataset\"");
            System.exit(1);
        }

        String sparkMaster = args[0];  // --- for cluster mode  e.g. "spark://afog-master:7077"
        //String sparkMaster = "local[*]"    //--- for local running
        //String sparkMaster = "spark://afog-master:7077"  //--- fixed the master node

        String modelPath = args[1];
        //String modelPath = "./src/main/resources/model/model.bin";
        //String modelPath = "hdfs://afog-master:9000/part4-projects/model/model.bin";  //SANG: for HDFS storage

        //--- Storage path ------------------
        String datafilePath = args[2];
        //String datafilePath = "hdfs://afog-master:9000/part4-projects/datasets/test_data.csv"; //SANG: for HDFS storage
        //String datafilePath = "./src/main/resources/datasets/test_data.csv";
        //String datafilePath = "./src/main/resources/datasets/test_data_small.csv";

        //------------------------------------------------
        // SANG: set log level
        Logger.getLogger("org.apache.spark").setLevel(Level.WARN);

        //===========================================================
        // 1. ---- Create a Spark Context/Session -----------------------
        SparkConf conf = new SparkConf();
        conf.setAppName("Model_Inference_with_DL4J_HDFS_Scale");
        //conf.setMaster("local[*]");
        conf.setMaster(sparkMaster);

        JavaSparkContext sc = new JavaSparkContext(conf);

        // 2. ---- Load pre-trained Keras-based model -------------------
        System.out.println("---- load pre-trained model ----------");

        //------------------ IMPORTANT -----------------------
        //SANG: to support read the model from HDFS!!!!
        //MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights(modelPath, true);  //--- Original work with .h5 model, but not support for Spark cluster and HDFS --> read Ref below
        //============================= ===========================
        //Ref: https://www.programmersought.com/article/83306085766/  --> in Scala, try to covert to Java
             //The spark cluster cannot directly load the h5 model file of Keras from hdfs
             // Other change from the Ref: from SparkComputationGraph --> restoreMultiLayerNetwork
        FileSystem fileSystem = FileSystem.get(sc.hadoopConfiguration()); //use this one or the one below --> Check on cluster
        //FileSystem fileSystem = FileSystem.get(new URI("hdfs://afog-master:9000"), sc.hadoopConfiguration());
        BufferedInputStream is = new BufferedInputStream(fileSystem.open(new Path(modelPath)));
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(is, false);  //SANG:

        //----------------------- --------------------------
        TrainingMaster tm = new ParameterAveragingTrainingMaster.Builder(1)
                .batchSizePerWorker(32)
                .averagingFrequency(5).workerPrefetchNumBatches(2)
                .build();

        // 3. ---- Create a network on Spark (sparkNet) -------------------
        System.out.println("------ Create a sparkNet  ---------");
        SparkDl4jMultiLayer sparkNet = new SparkDl4jMultiLayer(sc, model, tm);

        // 4. ------ Read the test dataset (testData) --------------------------------
        System.out.println("------ Read the test dataset  ---------");
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
        long startTime = System.nanoTime();

        System.out.println("------ Make prediction  ---------");
        //JavaPairRDD<String, INDArray> testPairs = testData.mapToPair(f-> new Tuple2("Occ_Predicted", f.getLabels() ));
        JavaPairRDD<String, INDArray> testPairs = testData.mapToPair(f-> new Tuple2("Predicted", f.getFeatures()));
        JavaPairRDD<String, INDArray> predictions = sparkNet.feedForwardWithKey(testPairs, 5);

        // --- output the predicted results ------
        /*List<DataSet> actualData = testData.collect(); //collect actual data
        AtomicInteger counter = new AtomicInteger(1);

        predictions.collect().forEach(p->{
            //System.out.println(p._1 + ": " + p._2); //simple print
            //System.out.println(p._1 + ": " + p._2.getInt()); // Print Int values in Int only

            ////Note: Ok reading data, but have problem with value converting with getLabels() ???
            //System.out.println("Unix-time:" + actualData.get(counter.get()).getLabels().getLong() + " ==> " + p._1 + ":" + p._2.getInt() + " - " + "Actual:" + actualData.get(counter.get()).getFeatures().getInt());  // start with AtomicInteger(0);
            //counter.getAndIncrement();

            System.out.println("Row:" + counter + " ==> " + p._1 + ":" + p._2.getInt() + " - " + "Actual:" + actualData.get(counter.getAndIncrement()-1).getFeatures().getInt());
        });*/


        // ---------------------------------------------
        System.out.println("\nTotal no. of records: " + predictions.count()); //Note: need a Spark action for all the lazy transformations above to be active
        System.out.println("********** DONE **********");

        long endTime = System.nanoTime();
        long duration = (endTime - startTime);
        System.out.println("\nRunning time (ms): " + duration/1000000); //divide by 1,000,000 to get milliseconds
        //System.out.println("Running time (s): " + duration/1000000000); //divide by 1,000,000,000 to get seconds

        // 7. save the predicted results in a file (in Hadoop text format part0000)
        //predictions.saveAsTextFile("./src/main/resources/inferences/");
    }

}
