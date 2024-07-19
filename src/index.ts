import { CSVLoader } from "@langchain/community/document_loaders/fs/csv";
import { SupabaseVectorStore } from "@langchain/community/vectorstores/supabase";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI, OpenAIEmbeddings } from "@langchain/openai";
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

const esimDocsOne = await loadCSVData("besims-one copy.csv");
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
      const retrievedData = await vectorStore.asRetriever().invoke(input);
      return {
        ...input,
        context: JSON.stringify(retrievedData)
      };
    },
    messages: new RunnablePassthrough(),
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

You are an avid traveler and an outgoing person who sells electronic SIM card.,

You are a friendly customer service representative for an eSIM traveler website. You welcome users and provide two options: product recommendations and a guide.

You will assist users through the purchase process by verifying device compatibility, destination, and duration, and then recommending appropriate eSIM products.

Emphasize clear and concise instructions, a friendly and helpful tone, quick responses, providing relevant recommendations, ensuring device compatibility, and guiding through the purchase process. Avoid technical jargon, lengthy or complicated responses, unrelated information, and pushing for a sale without proper guidance.

for device compatibility list you can use this data [device compatibility]

The interaction flow for recommendation is as follows: 
1. Welcome the user and offer two options: recommendations and a guide. 
2. Ask if the user wants to purchase an eSIM. 
3. If 'No', continue the guide discussion. 
4. If 'Yes', ask for the user's device model with examples. 
5. Check if the device supports eSIM. 
6. If not supported, inform the user and provide a URL to check supported models. 
7. If supported, ask for the user's destination and duration of stay. 
8. Search the database for suitable eSIM products based on the provided details. 
9. If no suitable product is found, apologize and ask if the user wants to search for other products. 
10. If suitable products are found, display the top 3 recommendations. 
11. Allow the user to choose a product and provide detailed information. 
12. Offer the option to continue to checkout or search for other products. 
13. If the user chooses to proceed, direct them to the checkout page.

The interaction flow for guide is as follows: 
1. Welcome the user and offer two options: recommendations and a guide. 
2. Ask if the user wants to see “guide”. 
3. If 'No', continue the recommendation discussion. 
4. If 'Yes', ask for “What can I help you?” and list of existing guides (About FAQ, Privacy Policy, terms and condition, and refund)
5. If user choose one of the option, you need explain based on data
6. At the end of explanation you need to give the menu about : 
  - contact admin → give contact admin
  - find other guide ? give the list option guide
  - try to purchase ? navigate purchase in our website

When additional information or clarification is needed, the chatbot will ask questions to ensure accuracy, such as: - 'Could you please provide more details about your device model?' - 'I didn't quite catch that. Could you specify the destination you'll be traveling to?' - 'Just to confirm, you'll be staying for [X] days at [destination], correct?

Maintain a friendly and welcoming tone, using phrases like: - 'Hi there! How can I assist you with your eSIM needs today?' - 'Great choice! Let's get you set up with the perfect eSIM for your trip.
  
You use Bahasa Indonesia and English as your main language. You will answer based on the user's language preference.

Input:

{input}`;

// make chunk of data with url to be used in the database
// error handle for no country code
// doest not to show recomendation if exact match found

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
    query = query.gte("duration_in_days", duration_in_days);
  }

  if (!data_unit && !data_amount && duration_in_days) {
    query = query.limit(2);
  }

  const { data, error } = await query;
  console.log(`log line 221: `, data);
  if(error){
    console.error(`Database error: ${error.message}`);
    return null
  }

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

// app.post("/prompt", async (c) => {
//   const body = await c.req.parseBody();

//   c.header("Content-Type", "text/event-stream");
//   c.header("Cache-Control", "no-cache");
//   c.header("Connection", "keep-alive");

//   // invoke the chain, passing user's query, setting up readable stream
//   const stream = new ReadableStream({
//     async start(controller) {
//       try {
//         await chain.invoke(String(body.text), {
//           callbacks: [
//             {
//               handleLLMNewToken(token: string) {
//                 controller.enqueue(`data: ${token}\n\n`);
//               },
//             },
//           ],
//         });

//         controller.enqueue("event: done\ndata: [DONE]\n\n");
//         controller.close();
//       } catch (error) {
//         controller.error(error);
//       }
//     },
//   });

//   return c.newResponse(stream);
// });



// app.post("/prompt", async (c: any) => {
//   try {
//     const body = await c.req.json();
//     const messages = body.messages ?? [];
//     const currentMessageContent = messages[messages.length - 1]?.content;

//     if (!currentMessageContent) {
//       return c.json({ error: "No messages or the last message does not have a content property" }, 500);
//     }

//     const prompt = PromptTemplate.fromTemplate(TEMPLATEPROMPT);
//     const schema = z.object({
//       country_code: z.string().describe("The code of the country"),
//       country_name: z.string().describe("The name of the country"),
//       data_amount: z.number().describe("The amount of data available"),
//       data_unit: z.enum(["MB", "GB"]).describe("The unit of data"),
//       duration_in_days: z.number().describe("The duration in days"),
//       chat_response: z
//         .string()
//         .describe(
//           "A response to the human's input and ask if they want to purchase that eSIM plan or not",
//         ),
//     });
//     const functionCallingModel = model.bind({
//       functions: [
//         {
//           name: "prompt_formatter",
//           description: "Should always be used to properly format prompt",
//           parameters: zodToJsonSchema(schema),
//         },
//       ],
//       function_call: { name: "prompt_formatter" },
//     });

//     try {

//       const chain = prompt
//         .pipe(functionCallingModel)
//         .pipe(new JsonOutputFunctionsParser());
          
//       const result: Document = await chain.invoke({ 
//         input: currentMessageContent,
//       });
//       console.log("Result:", result);

//       if(!result){
//         return c.json({ error: "No result" }, 404);
//       }

//       let { country_code, country_name, data_amount, data_unit, duration_in_days, chat_response } = result;
//       country_code = country_code ?? null;
//       country_name = country_name ?? null;
//       data_amount = data_amount ?? null;
//       data_unit = data_unit ?? null;
//       duration_in_days = duration_in_days ?? null;

//       const countryCode = await getCountryCode(country_name);
//       const esimData = await getEsimData(countryCode.code, data_amount, data_unit, duration_in_days);
      
//       if(!esimData){
//         return c.json(response(404, false, `No exact match found. Here are the closest matches based on ${country_name} esim.`, chat_response, []));
//       }
      
//       if(esimData.length === 0){
//         const recommendation = await getCountryCode(country_name);
//         if(!recommendation){
//           return c.json(response(404, false, `No exact match found.`, chat_response, []));
//         }

//         const {data: closeMatches, error: closeMatchesError} = await supabaseClient
//           .from("besim")
//           .select("*")
//           .eq("country_code", recommendation)
//           .order("duration_in_days", { ascending: false })
//           .limit(2);

//         if(closeMatchesError){
//           return c.json(response(500, false, closeMatchesError.message, chat_response, []));
//         }

//         if(closeMatches.length === 0){
//           return c.json(response(404, false, `No exact match found. Here are the closest matches based on ${country_name} esim.`, chat_response, []));
//         }

//         return c.json(response(200, false, `No exact match found. Here are the closest matches based on ${country_name} esim.`, chat_response, closeMatches));
//       }

//       let limitedEsimData = esimData.slice(0, 2);
//       return c.json(response(200, true, "Exact match found", chat_response, 
//         limitedEsimData.map(item => ({
//           country_code: item.country_code,
//           country_name : countryCode.name,
//           created_at: item.created_at,
//           data_amount: item.data_amount,
//           data_unit: item.data_unit,
//           duration_in_days: item.duration_in_days,
//           id: item.id,
//           idr_price: item.idr_price,
//           option_id: item.option_id,
//           plan_option: item.plan_option,
//           updated_at: item.updated_at,
//         }))
//       ));
//     } catch (e: any) {
//       return c.json({ error: e.message }, 500);
//     }
//   } catch (e: any) {
//     return c.json({ error: e.message }, 500);
//   }
  
// });

app.post("/prompt", async (c: any) => {
  try {
    const body = await c.req.json();
    const messages = body.messages ?? [];
    const currentMessageContent = messages[messages.length - 1]?.content;

    if (!currentMessageContent) {
      return c.json({ error: "No messages or the last message does not have a content property" }, 500);
    }

    c.header("Content-Type", "text/event-stream");
    c.header("Cache-Control", "no-cache");
    c.header("Connection", "keep-alive");

    const stream = new ReadableStream({
      async start(controller) {
        try {
          await chain.invoke(currentMessageContent, {
            callbacks: [
              {
                handleLLMNewToken(token: string) {
                  controller.enqueue(`data: ${token}\n\n`);
                },
              },
            ],
          });
  
          controller.enqueue("event: done\ndata: [DONE]\n\n");
          controller.close();
        } catch (error) {
          controller.error(error);
        }
      },
    });

    return c.newResponse(stream);

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
      const esimData = await getEsimData(countryCode.code, data_amount, data_unit, duration_in_days);
      
      if(!esimData){
        return c.json(response(404, false, `No exact match found. Here are the closest matches based on ${country_name} esim.`, chat_response, []));
      }
      
      if(esimData.length === 0){
        const recommendation = await getCountryCode(country_name);
        if(!recommendation){
          return c.json(response(404, false, `No exact match found.`, chat_response, []));
        }

        const {data: closeMatches, error: closeMatchesError} = await supabaseClient
          .from("besim")
          .select("*")
          .eq("country_code", recommendation)
          .order("duration_in_days", { ascending: false })
          .limit(2);

        if(closeMatchesError){
          return c.json(response(500, false, closeMatchesError.message, chat_response, []));
        }

        if(closeMatches.length === 0){
          return c.json(response(404, false, `No exact match found. Here are the closest matches based on ${country_name} esim.`, chat_response, []));
        }

        return c.json(response(200, false, `No exact match found. Here are the closest matches based on ${country_name} esim.`, chat_response, closeMatches));
      }

      let limitedEsimData = esimData.slice(0, 2);
      return c.json(response(200, true, "Exact match found", chat_response, 
        limitedEsimData.map(item => ({
          country_code: item.country_code,
          country_name : countryCode.name,
          created_at: item.created_at,
          data_amount: item.data_amount,
          data_unit: item.data_unit,
          duration_in_days: item.duration_in_days,
          id: item.id,
          idr_price: item.idr_price,
          option_id: item.option_id,
          plan_option: item.plan_option,
          updated_at: item.updated_at,
        }))
      ));
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