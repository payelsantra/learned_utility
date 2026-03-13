package retrieval;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Map;

public class ThetaLoader {

    public static Map<String, Float> loadTheta(String file) throws Exception {

        Map<String, Float> theta = new HashMap<>();

        BufferedReader br = new BufferedReader(new FileReader(file));

        String line;

        while ((line = br.readLine()) != null) {

            String[] parts = line.split("\\t");

            if (parts.length != 2)
                continue;

            String term = parts[0];
            float weight = Float.parseFloat(parts[1]);

            theta.put(term, weight);
        }

        br.close();

        System.out.println("Loaded theta size = " + theta.size());

        return theta;
    }
}