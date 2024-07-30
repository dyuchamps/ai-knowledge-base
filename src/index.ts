import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
import { BufferMemory } from 'langchain/memory';

import { createClient } from "@supabase/supabase-js";
import { Hono } from "hono";
import { cors } from "hono/cors";

//generative ui imports
import { StringOutputParser } from "@langchain/core/output_parsers";
import { JsonOutputFunctionsParser } from "@langchain/core/output_parsers/openai_functions";
import { PromptTemplate } from "@langchain/core/prompts";
import { RunnablePassthrough, RunnableSequence } from "@langchain/core/runnables";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

interface Document {
  country_code: string;
  country_name: string;
  data_amount: number | null;
  data_unit: string | null;
  duration_in_days: number | null;
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

// const esimDocsOne = await loadCSVData("besims.csv");
const esimDocsOne = await loadCSVData("beliesim_sample_product.csv");
const esimDocsTwo = await loadCSVData("besims-two.csv");
const esimDocsThree = await loadCSVData("besims-three.csv");
const esimDocsFour = await loadCSVData("besims-four.csv");
const countryDocs = await loadCSVData("countries.csv");

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
    "You are a friendly customer service representative here to assist users with product recommendations. You will help users through the purchase process with the context as follows: {context}. Use the data from the beliesim_sample_product.csv file, which provides information for Japan, China, South Korea, and Indonesia, including country name, duration in days, data amount, data unit, and price.",
  ],
  [
    "system",
    "Emphasize clear and concise instructions, a friendly and helpful tone, quick responses, providing relevant recommendations, ensuring device compatibility, and guiding through the purchase process. Avoid technical jargon, lengthy or complicated responses, unrelated information, and pushing for a sale without proper guidance.",
  ],
  [
    "system",
    "The interaction flow for recommendations is as follows: 1. Welcome the user and ask if the user wants to purchase an eSIM.2. If 'No', say 'Terima kasih telah menghubungi kami. Jika Anda memerlukan bantuan di lain waktu, jangan ragu untuk menghubungi kami. 3. If 'Yes', ask for the user's destination and duration of stay. 4. Search the database for suitable eSIM products based on the provided details. 5. If no suitable product is found, apologize and ask if the user wants to search for other products. 6. If suitable products are found, display the top 3 recommendations. 7. Allow the user to choose a product and provide detailed information. 8. Offer the option to continue to checkout or search for other products. 9. If the user chooses to proceed, direct them to the checkout page.",
  ],
  [
    "system",
    "Error Handling: If the checkout process encounters an issue, apologize, provide assistance, and suggest alternative steps or contacting support. End with a thank you message and offer further assistance if needed.",
  ],
  [
    "system",
    "When additional information or clarification is needed, ask questions to ensure accuracy, such as: - 'Could you please provide more details about your device model?' - 'I didn't quite catch that. Could you specify the destination you'll be traveling to?' - 'Just to confirm, you'll be staying for [X] days at [destination], correct?'",
  ],
  [
    "system",
    "Maintain a friendly and welcoming tone, using phrases like: - 'Hi there! How can I assist you with your eSIM needs today?' - 'Great choice! Let`s get you set up with the perfect eSIM for your trip.",
  ],
  [
    "system",
    "You only use Bahasa Indonesia as your main language. If the user asks using another language, tell them to use Bahasa Indonesia.",
  ],

  // new MessagesPlaceholder("history"),
  ["human", "{text}"],
]);

const model = new ChatOpenAI({
  apiKey: Bun.env.OPENAI_API_KEY,
  model: "gpt-3.5-turbo",
  temperature: 1.5,
  streaming: true,
});


const memory = new BufferMemory({returnMessages: true, memoryKey: "history"});

// store the data into the "documents" table, init supabase vector store instance with openai embedding model and db config args
const vectorStore = await SupabaseVectorStore.fromDocuments(
  // docs,
  [...esimDocsOne, ...countryDocs],
  new OpenAIEmbeddings({
    apiKey: Bun.env.OPENAI_API_KEY,
    model: "text-embedding-3-large",
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
  {
    context: async (input) => {
      const context = await memory.loadMemoryVariables(input)
      return context;
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

// make chunk of data with url to be used in the database
// error handle for no country code
// doest not to show recomendation if exact match found

const getCountryCode = async (country_name: string) => {
  const { data, error } = await supabaseClient
  .from("countries")
  .select("*")
  .eq("name", country_name);
  
  if(error){
    console.error(`Database error: ${error.message}`);
    return null
  }

  if(data.length === 0){
    return null;
  }
  return data[0];
};

const getEsimData = async (country_code: string | null, data_amount: number | null, data_unit: string | null, duration_in_days: number | null) => {
  let query = supabaseClient
    .from("besim")
    .select("*")
  
  if (country_code) {
    query = query.like("country_code", `%${country_code}%`);
  }

  if (data_unit) {
    query = query.like("data_unit", `%${data_unit}%`).eq("data_unit", data_unit);
  }

  if (data_amount) {
    query = query.gte("data_amount", data_amount);
  }

  if (duration_in_days) {
    query = query.eq("duration_in_days", duration_in_days);
  }

  if (!data_unit && !data_amount && duration_in_days) {
    query = query.limit(3);
  }

  const { data, error } = await query;

  if(error){
    console.error(`Database error: ${error.message}`);
    return null
  }

  console.log('line 248: ', data)
  return data;
};

const response = (status: number, success: boolean, message: string, chat_response: string, data: any) => {
  return {
    status,
    success,
    message,
    chat_response,
    data
  };
};

function sanitizeOutput(output: any) {
  return output.replace(/[^a-zA-Z0-9.,?!\s]/g, '');
}

const TEMPLATEPROMPT = `You are a friendly customer service representative here to assist users with product recommendations. 

You will help users through the purchase process with the context as follows. which provides information for Japan, China, South Korea, and Indonesia, including country name, duration in days, data amount, data unit, and price. 

Emphasize clear and concise instructions, a friendly and helpful tone, quick responses, relevant recommendations, ensuring guiding users through the purchase process. 

Avoid technical jargon, lengthy or complicated responses, unrelated information, and pushing for a sale without proper guidance. 

Do not hallucinate by retrieving all the data that only races to the supabase

The interaction flow for recommendations is as follows:
Welcome the user and ask if the user wants to purchase an eSIM.

If 'No', said "Terima kasih telah menghubungi kami. Jika Anda memerlukan bantuan di lain waktu, jangan ragu untuk menghubungi kami.

If yes, ask for the user's destination and duration of stay.

Search the database for suitable eSIM products based on the provided details.

If no suitable product is found, apologize and ask if the user wants to search for other products.

If suitable products are found, display recommendations.

Allow the user to choose a product and provide detailed information.
Offer the option to continue to checkout or search for other products.

If the user chooses to proceed, direct them to the checkout page.
Error Handling: If the checkout process encounters an issue, apologize, provide assistance, and suggest alternative steps or contacting support. End with a thank you message and offer further assistance if needed.

When additional information or clarification is needed, ask questions to ensure accuracy, such as: - 'Could you please provide more details about your device model?' - 'I didn't quite catch that. Could you specify the destination you'll be traveling to?' - 'Just to confirm, you'll be staying for [X] days at [destination], correct?'

Maintain a friendly and welcoming tone, using phrases like: - 'Hi there! How can I assist you with your eSIM needs today?' - 'Great choice! Let's get you set up with the perfect eSIM for your trip.'

You only use Bahasa Indonesia as your main language. If the user asks using another language, tell them to use Bahasa Indonesia.

Input:

{input}`;

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

      if(!result){
        return c.json({ error: "No result" }, 404);
      }

      let { country_code, country_name, data_amount, data_unit, duration_in_days, chat_response } = result;

      country_code = country_code ?? null;
      country_name = country_name ?? null;
      data_amount = data_amount ?? null;
      data_unit = data_unit ?? null;
      duration_in_days = duration_in_days ?? null;

      const countryCode = await getCountryCode(country_name);
      if(!countryCode){
        return c.json(response(404, false, `No exact match found.`, chat_response, []));
      }

      const esimData = await getEsimData(countryCode.code, data_amount, data_unit, duration_in_days);

      if(esimData?.length === 0){
        const {data: closeMatches, error: closeMatchesError} = await supabaseClient
          .from("besim")
          .select("*")
          .like("country_code", `%${countryCode.code}%`)
          .gte("duration_in_days", duration_in_days)
          .limit(3);

        if(closeMatchesError){
          return c.json(response(500, false, closeMatchesError.message, chat_response, []));
        }

        if(closeMatches.length === 0){
          return c.json(response(404, false, `No exact match found. Here are the closest matches based on ${country_name} esim.`, chat_response, []));
        }

        return c.json(response(400, false, `No exact match found. Here are the closest matches based on ${country_name} esim.`, chat_response, closeMatches));
      }

      return c.json(response(200, true, "Exact match found", chat_response, esimData));
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