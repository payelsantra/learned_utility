package retrieval;

import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.search.similarities.LMDirichletSimilarity;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.document.Document;

import java.io.*;
import java.nio.file.Paths;
import java.util.*;

public class UtilityOneStepRetrieverExtended {

    IndexReader reader;
    IndexSearcher searcher;

    Map<String,String> queries;
    Map<String,Float> theta;

    Map<String,Set<String>> qrels;
    Map<String,List<String>> answers;
    Set<String> yesNoQueries;

    UtilityQueryBuilder queryBuilder;

    BufferedWriter runWriter;
    BufferedWriter qrelsWriter;

    /* ---------------- CONSTRUCTOR ---------------- */

    public UtilityOneStepRetrieverExtended(
            String indexDir,
            String queryFile,
            String thetaFile,
            String qrelsFile,
            String answersFile,
            String runFile,
            String extendedQrelsFile) throws Exception {

        reader = DirectoryReader.open(
                FSDirectory.open(Paths.get(indexDir)));

        searcher = new IndexSearcher(reader);


//        searcher.setSimilarity(new BM25Similarity());
//        searcher.setSimilarity(
//                new BM25Similarity(Constants.BM25_K1, Constants.BM25_B)
//        );
        searcher.setSimilarity(
                new LMDirichletSimilarity(10)
        );

        queries = OneStepRetriever.loadQueries(queryFile);

        theta = ThetaLoader.loadTheta(thetaFile);

        System.out.println("Loaded theta size = " + theta.size());

        queryBuilder = new UtilityQueryBuilder(theta);

        qrels = loadQrels(qrelsFile);

        loadAnswers(answersFile);

        runWriter = new BufferedWriter(new FileWriter(runFile));

        qrelsWriter = new BufferedWriter(new FileWriter(extendedQrelsFile));

        copyOriginalQrels(qrelsFile);
    }

    /* ---------------- LOAD QRELS ---------------- */

    Map<String,Set<String>> loadQrels(String file) throws Exception {

        Map<String,Set<String>> map = new HashMap<>();

        BufferedReader br = new BufferedReader(new FileReader(file));

        String line;

        while((line = br.readLine()) != null){

            String[] parts = line.split("\\s+");

            String qid = parts[0];
            String docid = parts[2];

            map.computeIfAbsent(qid,k->new HashSet<>()).add(docid);
        }

        br.close();

        return map;
    }

    /* ---------------- COPY ORIGINAL QRELS ---------------- */

    void copyOriginalQrels(String file) throws Exception {

        BufferedReader br = new BufferedReader(new FileReader(file));

        String line;

        while((line = br.readLine()) != null){

            qrelsWriter.write(line);
            qrelsWriter.newLine();
        }

        br.close();
    }

    /* ---------------- LOAD ANSWERS ---------------- */

    void loadAnswers(String file) throws Exception {

        answers = new HashMap<>();
        yesNoQueries = new HashSet<>();

        BufferedReader br = new BufferedReader(new FileReader(file));

        String line;

        while((line = br.readLine()) != null){

            String id = line.split("\"id\":")[1].split("\"")[1];

            String ansPart = line.split("\"answers\":")[1];

            String arr = ansPart.split("\\[")[1].split("\\]")[0];

            String[] ans = arr.split(",");

            List<String> list = new ArrayList<>();

            for(String a:ans){

                a = a.replace("\"","").trim().toLowerCase();

                if(a.equals("yes") || a.equals("no")){
                    yesNoQueries.add(id);
                }

                list.add(a);
            }

            answers.put(id,list);
        }

        br.close();

        System.out.println("Answers loaded: " + answers.size());
    }

    /* ---------------- TOKEN MATCHING ---------------- */

    boolean containsAnswer(String text,String answer){

        text = text.toLowerCase();
        answer = answer.toLowerCase();

        text = text.replaceAll("[^a-z0-9 ]"," ");
        answer = answer.replaceAll("[^a-z0-9 ]"," ");

        String[] textTokens = text.split("\\s+");
        String[] ansTokens = answer.split("\\s+");

        if(ansTokens.length == 0)
            return false;

        for(int i=0;i<=textTokens.length-ansTokens.length;i++){

            boolean match=true;

            for(int j=0;j<ansTokens.length;j++){

                if(!textTokens[i+j].equals(ansTokens[j])){
                    match=false;
                    break;
                }
            }

            if(match)
                return true;
        }

        return false;
    }

    /* ---------------- RETRIEVE ---------------- */

    public void retrieve() throws Exception {

        System.out.println("Starting retrieval...");

        for(Map.Entry<String,String> e:queries.entrySet()){

            String qid = e.getKey();
            String qtext = e.getValue();

            Query query = queryBuilder.buildQuery(qtext);

            TopDocs hits = searcher.search(query,Constants.NUM_WANTED);

            processResults(qid,hits);
        }

        runWriter.close();
        qrelsWriter.close();

        System.out.println("Finished.");
    }

    /* ---------------- PROCESS RESULTS ---------------- */

    void processResults(String qid,TopDocs hits) throws Exception {

        boolean skipAnswerCheck = yesNoQueries.contains(qid);

        int rank=1;

        for(ScoreDoc sd:hits.scoreDocs){

            int docId = sd.doc;

            Document d = reader.document(docId);

            String docno = d.get(Constants.ID_FIELD);

            runWriter.write(String.format(
                    "%s\tQ0\t%s\t%d\t%.4f\tutility_run\n",
                    qid,docno,rank++,sd.score));

            if(skipAnswerCheck)
                continue;

            if(qrels.containsKey(qid) && qrels.get(qid).contains(docno))
                continue;

            checkAnswerAndExtend(qid,docno,docId);
        }
    }

    /* ---------------- EXTEND QRELS ---------------- */

    void checkAnswerAndExtend(String qid,String docno,int docId) throws Exception {

        if(!answers.containsKey(qid))
            return;

        Document doc = reader.document(docId);

        String text = doc.get(Constants.CONTENT_FIELD);

        if(text == null)
            return;

        for(String ans:answers.get(qid)){

            if(ans.equals("yes") || ans.equals("no"))
                continue;

            if(ans.length()<2)
                continue;

            if(containsAnswer(text,ans)){

                qrelsWriter.write(
                        qid + " 0 " + docno + " 1\n"
                );

                qrels.computeIfAbsent(qid,k->new HashSet<>()).add(docno);

                break;
            }
        }
    }

    /* ---------------- MAIN ---------------- */

    public static void main(String[] args) throws Exception {

        if(args.length < 7){

            System.out.println(
                    "Usage:\n" +
                            "java retrieval.UtilityOneStepRetrieverExtended\n" +
                            "<indexDir> <queryFile> <thetaFile> <qrelsFile> <answersFile> <runFile> <extendedQrelsFile>");

//            return;
//        }

        String BASE="/Users/payelsantra/Documents/PhD/Fusion-RAG/hotpotQA/";

            args=new String[]{
                    BASE+"index/",
                    BASE+"hotpotqa_validation_dpr.tsv",
                    BASE+"model/theta/withGT_theta_TF_infonce_post_Z_minmax.txt",
                    BASE+"qrels/hotpotqa_whole_qrels_validation.txt",
                    BASE+"hotpotqa_validation_dpr.jsonl",
                    BASE+"res/utility_valid_withGT_TF_infonce_post_Z_minmax_post_LMDIR_mu10.res",
                    BASE+"qrels/qrels/extended_qrels_valid_withGT_TF_infonce_post_Z_minmax_post_LMDIR_mu10.txt"
            };
        }

        UtilityOneStepRetrieverExtended r =
                new UtilityOneStepRetrieverExtended(
                        args[0],
                        args[1],
                        args[2],
                        args[3],
                        args[4],
                        args[5],
                        args[6]);

        r.retrieve();
    }
}
    /* ---------------- MAIN ---------------- */

//    public static void main(String[] args) throws Exception {
//
//        if(args.length < 7){
//
//            System.out.println(
//                    "Usage:\n" +
//                            "java UtilityOneStepRetrieverExtended\n" +
//                            "<indexDir> <queryFile> <thetaFile> <qrelsFile> <answersFile> <runFile> <extendedQrelsFile>");
//
//            String BASE="/Users/payelsantra/Documents/PhD/Fusion-RAG/hotpotQA/";
//
//            args=new String[]{
//                    BASE+"index/",
//                    BASE+"hotpotqa_validation_dpr.tsv",
//                    BASE+"model/theta/theta_TF_ridge_post_Z_minmax.txt",
//                    BASE+"qrels/hotpotqa_whole_qrels_valid.txt",
//                    BASE+"hotpotqa_validation_dpr.jsonl",
//                    BASE+"res/run.res",
//                    BASE+"qrels/extended_qrels.txt"
//            };
//        }
//
//        UtilityOneStepRetrieverExtended r =
//                new UtilityOneStepRetrieverExtended(
//                        args[0],
//                        args[1],
//                        args[2],
//                        args[3],
//                        args[4],
//                        args[5],
//                        args[6]);
//
//        r.retrieve();
//    }
//}