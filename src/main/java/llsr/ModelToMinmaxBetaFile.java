package llsr;

import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.store.FSDirectory;
import utils.IndexUtils;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;

public class ModelToMinmaxBetaFile {

    String modelFile;
    String indexDir;
    String outFile;

    Map<Integer, String> featureIdToTerm = new HashMap<>();

    public ModelToMinmaxBetaFile(String modelFile,
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
    }

    /**
     * Convert model weights → min-max normalized theta weights
     */
    void convertModelToBeta() throws Exception {

        BufferedReader br = new BufferedReader(new FileReader(modelFile));

        String line;
        boolean weightsStart = false;

        List<Float> weightList = new ArrayList<>();

        float minWeight = Float.MAX_VALUE;
        float maxWeight = -Float.MAX_VALUE;

        // -------- PASS 1: Read weights + find min/max --------
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

            if (weight < minWeight)
                minWeight = weight;

            if (weight > maxWeight)
                maxWeight = weight;
        }

        br.close();

        if (weightList.isEmpty()) {
            System.out.println("No weights found.");
            return;
        }

        System.out.println("Weight statistics:");
        System.out.println("Min weight = " + minWeight);
        System.out.println("Max weight = " + maxWeight);

        float range = maxWeight - minWeight;

        BufferedWriter bw = new BufferedWriter(new FileWriter(outFile));

        float minTheta = Float.MAX_VALUE;
        float maxTheta = -Float.MAX_VALUE;

        int featureId = 1;

        // -------- PASS 2: Normalize --------
        for (float weight : weightList) {

            float normalized = (weight - minWeight) / (range + 1e-6f);
            float theta = 1.0f + normalized;

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

        IndexUtils.init(searcher);
        IndexUtils.buildVocabulary();

        buildReverseVocabulary();

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

            args[0] = BASE + "model/model_ridge.txt";
            args[1] = BASE + "index/";
            args[2] = BASE + "model/theta/theta_ridge_post_minmax_scale.txt";
        }

        String modelFile = args[0];
        String indexDir = args[1];
        String outFile = args[2];

        ModelToBetaFile obj =
                new ModelToBetaFile(modelFile, indexDir, outFile);

        obj.process();
    }
}