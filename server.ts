import express from "express";
import path from "path";
import { GoogleGenAI, Type, GenerateContentResponse } from "@google/genai";
import { createServer as createViteServer } from "vite";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

// Allow large base64 screenshot uploads
app.use(express.json({ limit: "15mb" }));
app.use(express.urlencoded({ limit: "15mb", extended: true }));

// Lazy initialize Gemini client safely
let aiClient: GoogleGenAI | null = null;

function getGeminiClient(): GoogleGenAI {
  if (!aiClient) {
    const key = process.env.GEMINI_API_KEY;
    if (!key) {
      throw new Error("GEMINI_API_KEY is not defined in environment variables. Please configure it in Settings > Secrets.");
    }
    aiClient = new GoogleGenAI({
      apiKey: key,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

// API endpoint for trade prediction
app.post("/api/predict", async (req, res) => {
  try {
    const { image, additionalContext } = req.body;

    if (!image) {
      return res.status(400).json({ error: "No image screenshot provided" });
    }

    // Verify and clean base64 data
    const matches = image.match(/^data:([a-zA-Z0-9]+\/[a-zA-Z0-9-.+]+);base64,(.+)$/);
    let mimeType = "image/png";
    let base64Data = image;

    if (matches && matches.length === 3) {
      mimeType = matches[1];
      base64Data = matches[2];
    }

    const ai = getGeminiClient();

    // Prompts optimized for technical analysis of financial / trading charts
    const systemInstruction = `You are a professional financial market analyst and binary options trading expert specializing in candlestick analysis, price action, support/resistance, chart patterns, and technical indicators. Your job is to analyze Quotex platform screenshots and provide a detailed, accurate trade prediction. Always maintain objective, quantitative reasoning and explain exact visual cues from the screenshot.`;

    const promptText = `Analyze this Quotex screenshot chart in detail. Identify:
1. Current candlestick structures (e.g., trend, size, wick length, and specific patterns like Hammer, Engulfing, Doji, etc.).
2. Key support and resistance levels visible.
3. Market momentum (RSI, MACD, Moving Averages, Bollinger Bands, or general price velocity if indicators are visible).
4. Predict the most probable immediate trade direction: UP (Call/Buy), DOWN (Put/Sell), or HOLD (Wait for clear setup).

Additional context from user: ${additionalContext || "None provided"}

Provide the analysis strictly in JSON format as per the required schema. Ensure your reasons are technical, accurate, and completely avoid generic platitudes.`;

    const imagePart = {
      inlineData: {
        mimeType: mimeType,
        data: base64Data,
      },
    };

    const textPart = {
      text: promptText,
    };

    // Fallback models chain to handle 503 model high-demand/temporary failures elegantly
    const modelsToTry = ["gemini-3.5-flash", "gemini-2.5-flash", "gemini-2.5-pro"];
    let responseText = "";
    let lastError: any = null;

    for (const modelName of modelsToTry) {
      let attempts = 0;
      const maxAttempts = 2; // Try up to 2 times per model with a delay
      let succeeded = false;

      while (attempts < maxAttempts && !succeeded) {
        try {
          console.log(`Requesting prediction from model ${modelName} (Attempt ${attempts + 1}/${maxAttempts})`);
          
          const response = await ai.models.generateContent({
            model: modelName,
            contents: { parts: [imagePart, textPart] },
            config: {
              systemInstruction: systemInstruction,
              temperature: 0.2,
              responseMimeType: "application/json",
              responseSchema: {
                type: Type.OBJECT,
                required: [
                  "prediction",
                  "confidence",
                  "recommendedTimeframe",
                  "priceAction",
                  "keyPatterns",
                  "technicalIndicators",
                  "safeMargin",
                  "riskLevel",
                  "justification"
                ],
                properties: {
                  prediction: {
                    type: Type.STRING,
                    description: "The immediate trade prediction direction. Must be one of: 'UP', 'DOWN', or 'HOLD'.",
                  },
                  confidence: {
                    type: Type.INTEGER,
                    description: "The confidence rating of this prediction expressed as an integer percentage between 0 and 100.",
                  },
                  recommendedTimeframe: {
                    type: Type.STRING,
                    description: "The suggested trade duration based on the chart's candle timeframe (e.g., '1 Minute', '3 Minutes', '5 Minutes', etc.).",
                  },
                  priceAction: {
                    type: Type.STRING,
                    description: "Detailed analysis of support/resistance, trendlines, wicks, and general market structure identified.",
                  },
                  keyPatterns: {
                    type: Type.ARRAY,
                    items: { type: Type.STRING },
                    description: "List of key chart or candlestick patterns detected in the screenshot (e.g., 'Bullish Engulfing', 'Bearish Pinbar', 'Support Zone Rebound').",
                  },
                  technicalIndicators: {
                    type: Type.STRING,
                    description: "Evaluation of any technical indicators visible on the screenshot (e.g. RSI, Moving Averages, MACD, Bollinger Bands) or price momentum analysis.",
                  },
                  safeMargin: {
                    type: Type.STRING,
                    description: "Specific entry timing or price margin advice. For example: 'Enter on next candle touch of 1.09540 support' or 'Wait for a brief retest before entering'.",
                  },
                  riskLevel: {
                    type: Type.STRING,
                    description: "The level of risk associated with this prediction. Must be one of: 'LOW', 'MEDIUM', 'HIGH'.",
                  },
                  justification: {
                    type: Type.STRING,
                    description: "Clear and technical step-by-step reasoning that explains exactly why this prediction was made based on visual cues.",
                  },
                },
              },
            },
          });

          if (response && response.text) {
            responseText = response.text;
            succeeded = true;
            console.log(`Success with model ${modelName}!`);
            break;
          }
          throw new Error("Received empty response from the model.");

        } catch (err: any) {
          lastError = err;
          attempts++;
          const errMsg = err.message || "";
          const isTransient = errMsg.includes("503") || errMsg.includes("UNAVAILABLE") || errMsg.includes("demand") || errMsg.includes("overload");

          console.warn(`Model ${modelName} attempt ${attempts} failed:`, errMsg);

          if (attempts < maxAttempts && isTransient) {
            const backoffMs = attempts * 1500;
            console.log(`Transient model error detected. Waiting ${backoffMs}ms before retrying same model...`);
            await new Promise((resolve) => setTimeout(resolve, backoffMs));
          } else {
            // Break from the retry loop to fall back to the next model in the chain
            break;
          }
        }
      }

      if (succeeded) {
        break;
      }
    }

    if (!responseText) {
      throw lastError || new Error("All prediction models in the fallback chain were overloaded or unavailable. Please try again in a few seconds.");
    }

    const predictionResult = JSON.parse(responseText);
    res.json(predictionResult);

  } catch (error: any) {
    console.error("Prediction API Error:", error);
    res.status(500).json({
      error: error.message || "Failed to analyze Quotex screenshot. Ensure a clear chart image is provided.",
    });
  }
});

// Configure Vite or Serve Static Assets
async function initServer() {
  if (process.env.NODE_ENV !== "production") {
    console.log("Starting server in development mode with Vite middleware...");
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    console.log("Starting server in production mode...");
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Quotex Predictor Server running on http://localhost:${PORT}`);
  });
}

initServer().catch((err) => {
  console.error("Failed to start server:", err);
  process.exit(1);
});
