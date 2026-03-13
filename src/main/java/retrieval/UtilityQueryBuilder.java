package retrieval;

import indexing.MsMarcoIndexer;
import org.apache.lucene.index.Term;
import org.apache.lucene.search.*;
import retrieval.Constants;

import java.util.Map;

public class UtilityQueryBuilder {

    Map<String, Float> theta;

    public UtilityQueryBuilder(Map<String, Float> theta) {
        this.theta = theta;
    }

    public Query buildQuery(String queryText) throws Exception {

        BooleanQuery.Builder qb = new BooleanQuery.Builder();

        String[] tokens =
                MsMarcoIndexer.analyze(
                        MsMarcoIndexer.constructAnalyzer(),
                        queryText).split("\\s+");

        for (String token : tokens) {

            float weight = theta.getOrDefault(token, 1.0f);

            TermQuery tq =
                    new TermQuery(
                            new Term(Constants.CONTENT_FIELD, token));

//            System.out.println("weight: " + weight);
            Query boosted = new BoostQuery(tq, weight);

            qb.add(boosted, BooleanClause.Occur.SHOULD);
        }

        return qb.build();
    }
}