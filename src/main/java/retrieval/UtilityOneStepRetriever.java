package retrieval;

import indexing.MsMarcoIndexer;
import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.store.FSDirectory;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;

public class UtilityOneStepRetriever {

    IndexReader reader;
    IndexSearcher searcher;

    Map<String, String> queries;
    Map<String, Float> theta;

    String queryFile;
    String resFile;

    UtilityQueryBuilder queryBuilder;

    public UtilityOneStepRetriever(String indexDir,
                                   String queryFile,
                                   String thetaFile,
                                   String resFile) throws Exception {

        reader = DirectoryReader.open(
                FSDirectory.open(Paths.get(indexDir)));

        searcher = new IndexSearcher(reader);

//        searcher.setSimilarity(new BM25Similarity());
        searcher.setSimilarity(
                new BM25Similarity(Constants.BM25_K1, Constants.BM25_B)
        );
//        searcher.setSimilarity(
//                new LMDirichletSimilarity(50)
//        );

        this.queryFile = queryFile;
        this.resFile = resFile;


        queries = OneStepRetriever.loadQueries(queryFile);

        theta = ThetaLoader.loadTheta(thetaFile);

        System.out.println("Theta loaded: " + theta.size());

//        System.out.println("Sample theta weights:");
//
//        int count = 0;
//        for (Map.Entry<String,Float> e : theta.entrySet()) {
//
//            System.out.println(e.getKey() + " -> " + e.getValue());
//
//            if (++count == 10)
//                break;
//        }

        queryBuilder = new UtilityQueryBuilder(theta);
    }

    public void retrieve() throws Exception {

        BufferedWriter bw =
                new BufferedWriter(new FileWriter(resFile));

        System.out.println("Saving results to " + resFile);

        for (Map.Entry<String, String> e : queries.entrySet()) {

            String qid = e.getKey();
            String qtext = e.getValue();

            Query luceneQuery =
                    queryBuilder.buildQuery(qtext);

            TopDocs hits =
                    searcher.search(luceneQuery,
                            Constants.NUM_WANTED);
//            System.out.println("\nQuery: " + qtext);
//            System.out.println("Lucene Query: " + luceneQuery.toString());

            saveResults(bw, qid, hits);
        }

        bw.close();
    }

    void saveResults(BufferedWriter bw,
                     String qid,
                     TopDocs topDocs) throws Exception {

        int rank = 1;

        for (ScoreDoc sd : topDocs.scoreDocs) {

            int docId = sd.doc;

            String docno =
                    reader.document(docId)
                            .get(Constants.ID_FIELD);

            bw.write(String.format(
                    "%s\tQ0\t%s\t%d\t%.4f\tutility_run\n",
                    qid, docno, rank++, sd.score));
        }
    }

    public static void main(String[] args) throws Exception {

        if (args.length < 4) {

            String BASE =
                    "/Users/payelsantra/Documents/PhD/learned_utility/";

            args = new String[4];

            args[0] = BASE + "data/hotpotQA/index/";
            args[1] = BASE + "data/hotpotQA/base_data/hotpotqa_validation_dpr.tsv";
            args[2] = BASE + "data/hotpotQA/model/term_weights/theta_TF_infonce_post_Z_minmax.txt";
            args[3] = BASE + "data/hotpotQA/res_utility/utility_valid_TF_infonce.res";
        }

        UtilityOneStepRetriever retriever =
                new UtilityOneStepRetriever(
                        args[0],
                        args[1],
                        args[2],
                        args[3]);

        retriever.retrieve();
    }
}