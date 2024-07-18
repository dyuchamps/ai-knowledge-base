import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import {
  RunnablePassthrough,
  RunnableSequence,
} from "@langchain/core/runnables";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { createClient } from "@supabase/supabase-js";
import { Hono } from "hono";
import { cors } from "hono/cors";

//generative ui imports
import { JsonOutputFunctionsParser } from "@langchain/core/output_parsers/openai_functions";
import { PromptTemplate } from "@langchain/core/prompts";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

interface Document {
  country_code: string;
  country_name: string;
  data_amount: number;
  data_unit: string;
  duration_in_days: number;
  idr_price: number;
  chat_response: string;
}

interface Output {
  country_name: string;
  data_amount: number;
  data_unit: string;
  duration_in_days: number;
  price_in_usd: number;
  word_count: number;
  chat_response: string;
}

const app = new Hono();

const loadCSVData = async (filePath: string) => {
  const loader = new CSVLoader(filePath);
  return await loader.load();
};

const esimDocsOne = await loadCSVData("besims-one copy.csv");
const esimDocsTwo = await loadCSVData("besims-two.csv");
const esimDocsThree = await loadCSVData("besims-three.csv");
const esimDocsFour = await loadCSVData("besims-four.csv");
const countryDocs = await loadCSVData("countries.csv");

// const docs = await loader.load();

// init supabase instance
const supabaseClient = createClient(
  Bun.env.SUPABASE_URL!,
  Bun.env.SUPABASE_ANON_KEY!,
);

// delete any existing data from "documents" table, otherwise, each server starts the data will be duplicated
await supabaseClient.from("documents").delete().neq("id", 0);

const promptTemplate = ChatPromptTemplate.fromMessages([
  [
    "system",
    "You are an avid traveler and an outgoing person who sells electronic SIM card.",
  ],
  [
    "system",
    "You will assist your customer in buying their electronic SIM card plans with the context as follows: {context}",
  ],
  [
    "system",
    "If the customer tells you the duration of their travel or electronic SIM plan in format other than days, convert them to days first.",
  ],
  [
    "system",
    "If you happen to not have the electronic SIM card plan they request, apologize, and provide another electronic SIM card plans recommendation closest to what they requested but only based on the electronic SIM card plans you have. Maximum 2.",
  ],
  [
    "system",
    "If you have the electronic SIM card plan your customer requested, provide it with an exciting manner, and tell them to have a good trip. Only provide on most relevant answer.",
  ],
  [
    "system",
    "Extract the requested fields from the input. The field 'entity' refers to the first mentioned entity in the input."
  ],
  ["human", "{text}"],
]);

const model = new ChatOpenAI({
  apiKey: Bun.env.OPENAI_API_KEY,
  model: "gpt-3.5-turbo",
  temperature: 1.5,
  streaming: true,
});

// store the data into the "documents" table, init supabase vector store instance with openai embedding model and db config args
const vectorStore = await SupabaseVectorStore.fromDocuments(
  // docs,
  [...esimDocsOne, ...countryDocs],
  new OpenAIEmbeddings({
    apiKey: Bun.env.OPENAI_API_KEY,
    model: "text-embedding-ada-002",
  }),
  {
    client: supabaseClient,
    tableName: "documents",
    queryName: "match_documents",
  },
);

// runnable sequence (chaining events), provide context (including user's query/text) to/alongside the prompt template, define model, and an output parser
const chain = RunnableSequence.from([
  {
    context: async (input) => {
      return JSON.stringify(await vectorStore.asRetriever().invoke(input));
    },
    text: new RunnablePassthrough(),
  },
  promptTemplate,
  model,
  new StringOutputParser(),
]);

app.use(
  cors({
    origin: "*",
  }),
);

app.get("/", (c) => {
  return c.text("Hello Hono!");
});

const TEMPLATEPROMPT = `Extract the requested fields from the input.

The field "entity" refers to the first mentioned entity in the input.

If you have the electronic SIM card plan your customer requested, provide it with an exciting manner, and tell them to have a good trip. Only provide on most relevant answer.

If you happen to not have the electronic SIM card plan they request, apologize, and provide another electronic SIM card plans recommendation closest to what they requested but only based on the electronic SIM card plans you have. Maximum 2.

If the customer tells you the duration of their travel or electronic SIM plan in format other than days, convert them to days first.

"You are an avid traveler and an outgoing person who sells electronic SIM card.",


Input:

{input}`;


const getCountryCode = async (countryName: string) => {
  const { data, error } = await supabaseClient
  .from("countries")
  .select("*")
  .eq("name", countryName);
  
  if(error){
    console.error(`Database error: ${error.message}`);
    return null
  }

  if(data.length === 0){
    return null;
  }
  return data[0];
};

const getEsimData = async (country_code: string, data_amount: number, data_unit: string, duration_in_days: number) =>  {
  console.log(`Querying for country code: ${country_code}`);
  const { data, error } = await supabaseClient
    .from("besim")
    .select("*")
    .like("country_code", `%${country_code}%`)
    .like("data_unit", `%${data_unit}%`)
    .eq("data_unit", data_unit)
    .gte("data_amount", data_amount)
    .gte("duration_in_days", duration_in_days);

  if(error){
    console.error(`Database error: ${error.message}`);
    return null
  }

  return data;
}

app.post("/prompt", async (c: any) => {
  try {
    const body = await c.req.json();
    const messages = body.messages ?? [];
    const currentMessageContent = messages[messages.length - 1]?.content;

    if (!currentMessageContent) {
      return c.json({ error: "No messages or the last message does not have a content property" }, 500);
    }

    const prompt = PromptTemplate.fromTemplate(TEMPLATEPROMPT);
    const schema = z.object({
      country_code: z.string().describe("The code of the country"),
      country_name: z.string().describe("The name of the country"),
      data_amount: z.number().describe("The amount of data available"),
      data_unit: z.enum(["MB", "GB"]).describe("The unit of data"),
      duration_in_days: z.number().describe("The duration in days"),
      chat_response: z
        .string()
        .describe(
          "A response to the human's input and ask if they want to purchase that eSIM plan or not",
        ),
    });
    const functionCallingModel = model.bind({
      functions: [
        {
          name: "prompt_formatter",
          description: "Should always be used to properly format prompt",
          parameters: zodToJsonSchema(schema),
        },
      ],
      function_call: { name: "prompt_formatter" },
    });

    try {
      const chain = prompt
        .pipe(functionCallingModel)
        .pipe(new JsonOutputFunctionsParser());
      
      const result: Document = await chain.invoke({ 
        input: currentMessageContent,
      });
      console.log("Result:", result);

      const { country_name, data_amount, data_unit, duration_in_days } = result;

      const countryCode = await getCountryCode(country_name);
      const esimData = await getEsimData(countryCode.code, data_amount, data_unit, duration_in_days);
      console.log(`Esim data: ${JSON.stringify(esimData, null, 2)}`);
      let responseJson;
      if(!esimData){
        return responseJson = {
          status: 404,
          success: false,
          result: `No exact match found. Here are the closest matches based on ${country_name} esim.`,
        };
      }
      

      if(esimData.length === 0){
        const recommendation = await getCountryCode(country_name);
        const {data: closeMatches, error: closeMatchesError} = await supabaseClient
          .from("besim")
          .select("*")
          .eq("country_code", recommendation)
          .order("duration_in_days", { ascending: false })
          .limit(2);

        if(closeMatchesError){
          return c.json({ error: closeMatchesError.message }, 500);
        }

        responseJson = {
          status: 200,
          success: false,
          message: `No exact match found. Here are the closest matches based on ${country_name} esim.`,
          chat_response: result.chat_response,
          data: closeMatches,
        };
      } else {
        responseJson = {
          status: 200,
          success: true,
          messages: `Exact match found for country: ${country_name}`,
          chat_response: result.chat_response,
          data: {
            country_code: esimData[0].country_code,
            country_name : countryCode.name,
            created_at: esimData[0].created_at,
            data_amount: esimData[0].data_amount,
            data_unit: esimData[0].data_unit,
            duration_in_days: esimData[0].duration_in_days,
            id: esimData[0].id,
            idr_price: esimData[0].idr_price,
            option_id: esimData[0].option_id,
            plan_option: esimData[0].plan_option,
            updated_at: esimData[0].updated_at,
          }
          
        }
      }

      return c.json(responseJson, 200);
    } catch (e: any) {
      return c.json({ error: e.message }, 500);
    }
  } catch (e: any) {
    return c.json({ error: e.message }, 500);
  }
  
});

// Define the model to extract the requested fields from the input
const TEMPLATE = `Extract the requested fields from the input.

The field "entity" refers to the first mentioned entity in the input.

Input:

{input}`;

app.post("/structured_output", async (c) => {
  try {
    const body = await c.req.json();
    const messages = body.messages ?? [];
    const currentMessageContent = messages[messages.length - 1].content;

    const prompt = PromptTemplate.fromTemplate(TEMPLATE);

    // Define the schema for the output
    const schema = z.object({
      country_name: z.string().describe("The name of the country"),
      data_amount: z.number().describe("The amount of data available"),
      data_unit: z.enum(["mb", "gb"]).describe("The unit of data"),
      duration_in_days: z.number().describe("The duration in days"),
      price_in_usd: z.number().describe("The price in USD"),
      word_count: z.number().describe("The number of words in the input"),
      chat_response: z
        .string()
        .describe(
          "A response to the human's input and ask if they want to purchase that eSIM plan or not",
        ),
    });

    // Define the model to call the output_formatter function
    const functionCallingModel = model.bind({
      functions: [
        {
          name: "output_formatter",
          description: "Should always be used to properly format output",
          parameters: zodToJsonSchema(schema),
        },
      ],
      function_call: { name: "output_formatter" },
    });

    // Call the model to extract the requested fields from the input
    try {
      const chain = prompt
        .pipe(functionCallingModel)
        .pipe(new JsonOutputFunctionsParser());
      const result: Output = await chain.invoke({
        input: currentMessageContent,
      });

      console.log("Result:", result);

      // Extract the country name from the result
      const { country_name, duration_in_days } = result;
      console.log(`Querying for country: ${country_name}`);
      console.log(`Duration in days: ${duration_in_days}`);

      // Query the Supabase database for the full data based on the country name
      const { data, error } = await supabaseClient
        .from("esims")
        .select("*")
        .eq("country_name", country_name.toLowerCase());

      if (error) {
        console.error(`Database error: ${error.message}`);
        return c.json({ error: error.message }, 500);
      }

      if (data.length === 0) {
        console.log(`No data found for country: ${country_name}`);
        return c.json({ error: "No data found" }, 404);
      }

      console.log(`Data retrieved: `, data);

      // Assuming you want to return the first match
      const mergedResult = { ...result, ...data[0] };

      return c.json(mergedResult, 200);
    } catch (e: any) {
      console.error(`Unhandled error: ${e.message}`);
      return c.json({ error: e.message }, 500);
    }
  } catch (e: any) {
    console.error(`Unhandled error: ${e.message}`);
    return c.json({ error: e.message }, 500);
  }
});

export default app;