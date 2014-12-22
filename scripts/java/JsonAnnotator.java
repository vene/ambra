import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.charset.Charset;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Properties;

import javax.json.Json;
import javax.json.JsonArray;
import javax.json.JsonArrayBuilder;
import javax.json.JsonObject;
import javax.json.JsonObjectBuilder;
import javax.json.JsonReader;
import javax.json.JsonValue;
import javax.json.JsonWriter;

import edu.stanford.nlp.ling.CoreAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.LemmaAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.NamedEntityTagAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.PartOfSpeechAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.SentencesAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TextAnnotation;
import edu.stanford.nlp.ling.CoreAnnotations.TokensAnnotation;
import edu.stanford.nlp.ling.CoreLabel;
import edu.stanford.nlp.pipeline.Annotation;
import edu.stanford.nlp.pipeline.StanfordCoreNLP;
import edu.stanford.nlp.util.CoreMap;

/**
 * @author Alina Maria Ciobanu
 */
public class JsonAnnotator 
{	
	/**
	 * Adds tokens, lemmas, POS and NER annotations for the specified field of the json file.
	 *  
	 * @param inFile input json file
	 * @param outFile output json file
	 * @param key the key whose value in the json file will be annotated
	 */
	public void annotate(String inFile, String outFile, String key)
	{
		try
		{
			JsonArrayBuilder arrayBuilder = Json.createArrayBuilder();
			JsonReader jsonReader = Json.createReader(new BufferedReader(new InputStreamReader(new FileInputStream(new File(inFile)), Charset.forName("UTF8"))));
			JsonArray inArray = jsonReader.readArray();
			Iterator<JsonValue> it = inArray.iterator();
			
			Properties props = new Properties();
		        props.put("annotators", "tokenize, ssplit, pos, lemma, ner");
		        props.put("pos.model", "edu/stanford/nlp/models/pos-tagger/english-left3words/english-left3words-distsim.tagger");
		        props.put("ner.model", "edu/stanford/nlp/models/ner/english.muc.7class.distsim.crf.ser.gz");
		        StanfordCoreNLP pipeline = new StanfordCoreNLP(props);
		    			
			while (it.hasNext())
			{					
				JsonObject jsonObject = (JsonObject)it.next();
				JsonObjectBuilder builder = Json.createObjectBuilder();

				Annotation text = new Annotation(jsonObject.getString(key));
				pipeline.annotate(text);

				builder.add(key, jsonObject.get(key));
				
				addAnnotations(builder, text);
				
				for (Entry<String, JsonValue> entry : jsonObject.entrySet())
					if (!key.equals(entry.getKey()))
						builder.add(entry.getKey(), entry.getValue());
				
				arrayBuilder.add(builder.build());
			}
			
			JsonWriter jsonWriter = Json.createWriter(new BufferedWriter(new OutputStreamWriter(new FileOutputStream(new File(outFile)), Charset.forName("UTF8"))));
			jsonWriter.writeArray(arrayBuilder.build());			
			jsonWriter.close();
			jsonReader.close();
		}
		catch (IOException e)
		{
			System.out.println(e.getMessage());
		}
	}
	
	/**
	 * Adds tokens, lemmas, POS and NER annotations to the json builder.
	 * 
	 * @param builder the json builder to which the annotations are added
	 * @param text the span of text to be parsed 
	 */
	public void addAnnotations(JsonObjectBuilder builder, Annotation text)
	{
		Map<String, Class<? extends CoreAnnotation<String>>> annotators = new HashMap<String, Class<? extends CoreAnnotation<String>>>();
		
		annotators.put("tokens", TextAnnotation.class);
		annotators.put("pos", PartOfSpeechAnnotation.class);
		annotators.put("ner", NamedEntityTagAnnotation.class);
		annotators.put("lemmas", LemmaAnnotation.class);
		
		Map<String, JsonArrayBuilder> annotations = addAnnotations(text, annotators);

		for (String key : Arrays.asList("tokens", "pos", "ner", "lemmas"))
			builder.add(key, annotations.get(key).build());
	}
	
	/**
	 * Returns a map of json array builders for the given annotations. A key in the map represent the 
	 * name from the name/value pair in the json that will be created.
	 * 
	 * @param text the span of text to be parsed 
	 * @param annotators the annotators to be applied
	 * @return a map of json array builders for the given annotations
	 */
	public Map<String, JsonArrayBuilder> addAnnotations(Annotation text, Map<String, Class<? extends CoreAnnotation<String>>> annotators) 
	{
		List<CoreMap> sentences = text.get(SentencesAnnotation.class);

		Map<String, JsonArrayBuilder> documentBuilders = new HashMap<String, JsonArrayBuilder>();
		for (String key : annotators.keySet())
			documentBuilders.put(key, Json.createArrayBuilder());
		
		for(CoreMap sentence: sentences)
		{
			Map<String, JsonArrayBuilder> sentenceBuilders = new HashMap<String, JsonArrayBuilder>();
			for (String key : annotators.keySet())
				sentenceBuilders.put(key, Json.createArrayBuilder());
			
			for (CoreLabel token: sentence.get(TokensAnnotation.class)) 
	    	    for (String key : annotators.keySet())
	    	    	sentenceBuilders.get(key).add(token.get(annotators.get(key)));
			
			for (String key : annotators.keySet())
				documentBuilders.get(key).add(sentenceBuilders.get(key).build());
		}
		
		return documentBuilders;
	}

	public static void main(String[] args) 
	{
		if (args.length < 2 || args.length > 3) 
		{
			// TODO print usage
			System.out.println("Parameters: <input file> <output file> [<json key>]");
            		System.exit(-1);
        	}
		
		JsonAnnotator jsonAnnotator = new JsonAnnotator();
		jsonAnnotator.annotate(args[0], args[1], args.length == 3 ? args[2] : "text");
	}
}
