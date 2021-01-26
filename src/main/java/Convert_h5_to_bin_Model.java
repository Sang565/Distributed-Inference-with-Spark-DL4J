/*
 Purpose: Convert .h5 model to .bin for running on Spark cluster

*/

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.nn.modelimport.keras.KerasModelImport;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;

public class Convert_h5_to_bin_Model {

    public static <E> void main(String[] args) throws Exception {
        // SANG: set log level
        Logger.getLogger("org.apache.spark").setLevel(Level.WARN);

        // 1. ---- Create a Spark Context/Session -----------------------
        SparkConf conf = new SparkConf();
        conf.setAppName("Convert_h5_to_bin_Model");
        conf.setMaster("local[*]");

        JavaSparkContext sc = new JavaSparkContext(conf);

        // 2. ---- Load pre-trained Keras-based model -------------------
        System.out.println("---- load pre-trained model ----------");
        //String modelPath = "hdfs://afog-master:9000/part4-projects/model/model.h5";  //SANG: for HDFS storage
        //String localModelPath = "./src/main/resources/model/model.h5";
        MultiLayerNetwork model = KerasModelImport.importKerasSequentialModelAndWeights("./src/main/resources/model/model.h5", true);

        System.out.println("---- convert .h5 to .bin model ----------");
        //ModelSerializer.writeModel(model, "model.bin");
        ModelSerializer.writeModel(model, "./src/main/resources/model/model.bin", true);

        System.out.println("********** DONE **********");
    }
}
