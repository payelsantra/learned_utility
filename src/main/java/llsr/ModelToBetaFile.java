package llsr;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.FSDirectory;
import utils.IndexUtils;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;

public class ModelToBetaFile {

    String modelFile;
    String indexDir;
    String outFile;

    Map<Integer, String> featureIdToTerm = new HashMap<>();

    public ModelToBetaFile(String modelFile,
                           String indexDir,
                           String outFile) {
        this.modelFile = modelFile;
        this.indexDir = indexDir;
        this.outFile = outFile;
    }

    /**
     * Build reverse vocabulary mapping: featureId -> term
     */
    void buildReverseVocabulary() {

        for (Map.Entry<String, Integer> entry :
                IndexUtils.termToFeatureId.entrySet()) {

            featureIdToTerm.put(entry.getValue(), entry.getKey());
        }

        System.out.println("Reverse vocabulary size = "
                + featureIdToTerm.size());
        // ---- SANITY CHECK ----
        System.out.println("Sample featureId -> term mapping:");
        for (int i = 2122908; i <= 2122918; i++) {
            System.out.println(i + " -> " + featureIdToTerm.get(i));
        }
    }

    /**
     * Convert LIBLINEAR model weights → normalized theta weights
     */
    void convertModelToBeta() throws Exception {

        BufferedReader br = new BufferedReader(new FileReader(modelFile));

        String line;
        boolean weightsStart = false;

        List<Float> weightList = new ArrayList<>();

        // -------- PASS 1: Read raw model weights --------
        while ((line = br.readLine()) != null) {

            line = line.trim();

            if (line.equals("w")) {
                weightsStart = true;
                continue;
            }

            if (!weightsStart)
                continue;

            float weight = Float.parseFloat(line);
            weightList.add(weight);
        }

        br.close();

        if (weightList.isEmpty()) {
            System.out.println("No weights found.");
            return;
        }

        System.out.println("Sanity check: first few model weights with terms");

        for (int i = 2122908; i < 2122918 && i < weightList.size(); i++) {
            String term = featureIdToTerm.get(i + 1);
            float w = weightList.get(i);

            System.out.println((i+1) + " -> " + term + " : " + w);
        }

        // -------- Compute mean --------
        float sum = 0f;

        for (float w : weightList)
            sum += w;

        float mean = sum / weightList.size();

        // -------- Compute std --------
        float sqSum = 0f;

        for (float w : weightList)
            sqSum += (w - mean) * (w - mean);

        float std = (float) Math.sqrt(sqSum / weightList.size());

        System.out.println("Beta statistics:");
        System.out.println("Mean = " + mean);
        System.out.println("Std  = " + std);

        // -------- PASS 2: Compute Z-scores --------
        List<Float> zList = new ArrayList<>();

        float minZ = Float.MAX_VALUE;
        float maxZ = -Float.MAX_VALUE;

        for (float weight : weightList) {

            float z = (weight - mean) / (std + 1e-6f);

            zList.add(z);

            if (z < minZ)
                minZ = z;

            if (z > maxZ)
                maxZ = z;
        }

        System.out.println("Z-score statistics:");
        System.out.println("Min Z = " + minZ);
        System.out.println("Max Z = " + maxZ);

        // -------- PASS 3: Min-Max normalization --------

        BufferedWriter bw = new BufferedWriter(new FileWriter(outFile));

        float range = maxZ - minZ;

        float minTheta = Float.MAX_VALUE;
        float maxTheta = -Float.MAX_VALUE;

        int featureId = 1;

        for (float z : zList) {

            // Min-Max normalization
            float normalized = (z - minZ) / (range + 1e-6f);

            // shift to positive range [0.5 , 1.5]
            float theta = 0.5f + normalized;

            if (theta < minTheta)
                minTheta = theta;

            if (theta > maxTheta)
                maxTheta = theta;

            String term = featureIdToTerm.get(featureId);

            if (term != null) {
                bw.write(term + "\t" + theta);
                bw.newLine();
            }

            featureId++;
        }

        bw.close();

        System.out.println("Theta statistics after normalization:");
        System.out.println("Min Theta = " + minTheta);
        System.out.println("Max Theta = " + maxTheta);

        System.out.println("Theta file written to: " + outFile);
    }


    /**
     * Main processing pipeline
     */
    void process() throws Exception {

        System.out.println("Loading index from: " + indexDir);

        IndexSearcher searcher =
                new IndexSearcher(
                        DirectoryReader.open(
                                FSDirectory.open(
                                        Paths.get(indexDir))));

        // initialize statistics
        IndexUtils.init(searcher);
        IndexUtils.buildVocabulary();

        // reverse mapping
        buildReverseVocabulary();

        // convert model
        convertModelToBeta();

        System.out.println("Done.");
    }


    /**
     * MAIN
     */
    public static void main(String[] args) throws Exception {

        if (args.length < 3) {

            System.out.println("Using default arguments...");

            String BASE =
                    "/Users/payelsantra/Documents/PhD/Fusion-RAG/hotpotQA/";

            args = new String[3];

            args[0] = BASE + "model/withGT_model_TF_infonce.txt";
            args[1] = BASE + "index/";
            args[2] = BASE + "model/theta/withGT_theta_TF_infonce_post_Z_minmax.txt";
        }

        String modelFile = args[0];
        String indexDir = args[1];
        String outFile = args[2];

        ModelToBetaFile obj =
                new ModelToBetaFile(modelFile, indexDir, outFile);

        obj.process();
    }
}

//awk '$1==2122913 {print $2}' vocab_debug.txt