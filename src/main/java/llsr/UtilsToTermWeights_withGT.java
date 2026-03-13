package llsr;

import org.apache.lucene.index.*;
import org.apache.lucene.search.*;
import org.apache.lucene.search.similarities.BM25Similarity;
import org.apache.lucene.store.FSDirectory;
import org.apache.lucene.util.BytesRef;

import org.json.JSONObject;
import org.json.JSONArray;

import retrieval.Constants;
import retrieval.MsMarcoQuery;
import retrieval.OneStepRetriever;
import utils.IndexUtils;

import java.io.*;
import java.util.*;

public class UtilsToTermWeights_withGT {

    Map<String,String> queryMap;
    Map<String,String> answerMap;

    String utilsFile;
    String queryFile;
    String answerFile;
    String indexDir;
    String outFile;

    IndexReader reader;
    IndexSearcher searcher;

    UtilsToTermWeights_withGT(String[] args) throws Exception {

        utilsFile = args[0];
        queryFile = args[1];
        indexDir  = args[2];
        answerFile = args[3];
        outFile   = args[4];

        reader = DirectoryReader.open(
                FSDirectory.open(new File(indexDir).toPath()));

        searcher = new IndexSearcher(reader);
        searcher.setSimilarity(new BM25Similarity());

        IndexUtils.init(searcher);
        IndexUtils.buildVocabulary();
        queryMap = OneStepRetriever.loadQueries(queryFile);
        answerMap = loadAnswers(answerFile);
    }

    /* ---------------------------------------------
       LOAD ANSWERS FROM JSONL
       --------------------------------------------- */

    Map<String,String> loadAnswers(String answerFile) throws Exception {

        Map<String,String> map = new HashMap<>();

        BufferedReader br = new BufferedReader(new FileReader(answerFile));
        String line;

        while((line = br.readLine()) != null){
            JSONObject obj = new JSONObject(line);
            String qid = obj.getString("id");

            JSONArray answers = obj.getJSONArray("answers");

            if(answers.length() > 0){
                String ans = answers.getString(0);
                map.put(qid, ans);
            }
        }
        br.close();
        return map;
    }

    /* ---------------------------------------------
       EXTRACT QUERY TERMS
       --------------------------------------------- */

    Set<String> extractQueryTerms(Query query){

        Set<String> terms = new HashSet<>();

        query.visit(new QueryVisitor(){

            @Override
            public boolean acceptField(String field){
                return field.equals(Constants.CONTENT_FIELD);
            }
            @Override
            public void consumeTerms(Query q, Term... queryTerms){
                for(Term t : queryTerms)
                    terms.add(t.text());
            }
        });

        return terms;
    }

    /* ---------------------------------------------
       TERM FEATURE FUNCTIONS
       --------------------------------------------- */

    float computeBM25TermScore(int tf, int df, int docLen) {
        float k1 = 1.2f;
        float b = 0.75f;

        float idf = (float) Math.log(1.0 + (IndexUtils.N - df + 0.5) / (df + 0.5));

        float denom = tf + k1 * (1 - b + b * docLen / IndexUtils.avgdl);
        float tfWeight = tf * (k1 + 1) / denom;

        return idf * tfWeight;
    }

    float computeTF(int tf) {
        return (float) tf;
    }

    float computeTFIDF(int tf, int df) {

        float idf = (float) Math.log((IndexUtils.N + 1.0) / (df + 1.0));

        return tf * idf;
    }

    /* ---------------------------------------------
       PROCESS QUERY
       --------------------------------------------- */

    List<String> processQuery(String queryId,
                              List<ScoreDoc> scoreDocs)
            throws IOException {

        List<String> lines = new ArrayList<>();

        String queryText = queryMap.get(queryId);
        if(queryText == null)
            return lines;
        MsMarcoQuery query = new MsMarcoQuery(queryId, queryText);
        Query luceneQuery = query.getQuery();

        /* -------- FEATURE TERMS -------- */

        Set<String> featureTerms =
                new HashSet<>(extractQueryTerms(luceneQuery));

        String answerText = answerMap.get(queryId);

        if(answerText != null){

            String analyzed =
                    indexing.MsMarcoIndexer.analyze(
                            indexing.MsMarcoIndexer.constructAnalyzer(),
                            answerText);

            String[] ansTokens =
                    analyzed.split("\\s+");

            featureTerms.addAll(
                    Arrays.asList(ansTokens));
        }

        /* -------- PROCESS DOCS -------- */

        for(ScoreDoc sd : scoreDocs){

            int docId = sd.doc;
            float utility = sd.score;

            Terms tv = reader.getTermVector( docId,Constants.CONTENT_FIELD);

            if(tv == null)
                continue;

            TermsEnum te = tv.iterator();
            int docLen = 0;
            BytesRef term;
            while((term = te.next()) != null){
                PostingsEnum pe = te.postings(null);
                pe.nextDoc();
                docLen += pe.freq();
            }

            if(docLen == 0)
                continue;

            Map<Integer,Float> featureMap = new TreeMap<>();

            te = tv.iterator();
            for(String termStr : featureTerms){
                BytesRef br = new BytesRef(termStr);
                if(te.seekExact(br)){
                    PostingsEnum pe = te.postings(null);
                    pe.nextDoc();
                    int tf = pe.freq();
                    int df =  reader.docFreq(
                                    new Term(Constants.CONTENT_FIELD,termStr));
                    // ---------- SELECT FEATURE TYPE ----------
                    // Option 1: BM25
//                    float termScore = computeBM25TermScore(tf, df, docLen);
                    // Option 2: TF
                    float termScore = computeTF(tf);
                    // Option 3: TF-IDF
//                    float termScore = computeTFIDF(tf, df);
                    Integer featureId =IndexUtils.termToFeatureId.get(termStr);
                    if(featureId != null)
                        featureMap.put(featureId, termScore);
                }
            }

            if(featureMap.isEmpty())
                continue;

            StringBuilder sb =
                    new StringBuilder();

            sb.append(utility);

            for(Map.Entry<Integer,Float> e :
                    featureMap.entrySet()){

                sb.append(" ")
                        .append(e.getKey())
                        .append(":")
                        .append(e.getValue());
            }
            lines.add(sb.toString());
        }
        return lines;
    }

    /* ---------------------------------------------
       WRITE LINES
       --------------------------------------------- */

    void writeLines(BufferedWriter bw,
                    List<String> lines)
            throws Exception {

        for(String line : lines){
            bw.write(line);
            bw.newLine();
        }
    }

    /* ---------------------------------------------
       MAIN PROCESS
       --------------------------------------------- */

    void process() throws Exception {

        BufferedWriter bw =new BufferedWriter(new FileWriter(outFile));
        BufferedReader br =new BufferedReader(new FileReader(utilsFile));
        String line;
        List<ScoreDoc> sdList =new ArrayList<>();
        br.readLine(); // skip header
        String queryId = null;
        String prevQuery = null;

        int numQueries = 0;
        while((line = br.readLine()) != null){

            String[] fields =
                    line.split("\t");

            if(fields.length < 3){
                System.out.println("Skipping malformed row: " + line);
                continue;
            }
            queryId = fields[0];

            if(prevQuery != null && !prevQuery.equals(queryId)){
                writeLines(bw,processQuery(prevQuery,sdList));
                sdList.clear();
                numQueries++;
            }

            String docName = fields[1];
            int docId =IndexUtils.getDocOffsetFromId(docName);
            float utilValue =Float.parseFloat(fields[2]);
            sdList.add(new ScoreDoc(docId, utilValue));
            prevQuery = queryId;
        }

        writeLines(bw, processQuery(prevQuery, sdList));
        bw.close();
        System.out.println(
                "Processed " + numQueries + " queries");
    }

    /* ---------------------------------------------
       MAIN
       --------------------------------------------- */

    public static void main(String[] args)
            throws Exception {

        if(args.length < 5){
            System.out.println(
                    "usage: java UtilsToTermWeights_withGT "
                            + "<utils file> <query file> "
                            + "<index dir> <answers jsonl> "
                            + "<output file>");

            String BASE = "/Users/payelsantra/Documents/PhD/Fusion-RAG/hotpotQA/";

            args = new String[5];
            args[0] = BASE + "utilityWise_infonce_training_pairs.tsv";
            args[1] =  BASE + "hotpotqa_train_dpr.tsv";
            args[2] = BASE + "index/";
            args[3] = BASE + "hotpotqa_train_dpr.jsonl";
            args[4] = BASE + "withGT_termwts_TF_train_infonce.txt";
        }

        UtilsToTermWeights_withGT obj =
                new UtilsToTermWeights_withGT(args);

        obj.process();
    }
}