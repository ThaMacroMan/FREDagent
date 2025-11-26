# Raw Agent Description Standard (Executive Summary)

# Overview

The FRED Economic Data Agent is an AI-powered service that provides instant, comprehensive analysis of Federal Reserve Economic Data (FRED). Users submit natural language queries about economic indicators (unemployment, inflation, GDP, etc.) and receive expert-level analysis with calculated metrics, historical context, and actionable insights—all in 30-60 seconds via a simple API integrated with Cardano blockchain payments.

---

## 1. The Problem

Accessing and analyzing FRED economic data requires navigating complex databases, identifying correct series IDs, retrieving historical data, performing statistical calculations, and interpreting results—a time-consuming process requiring economic expertise.

---

## 2. The Solution

The agent uses CrewAI with two specialized agents: a FRED Data Analyst that searches and retrieves economic data with comprehensive statistical analysis, and an Economic Advisor that transforms raw data into structured insights. The system includes intelligent fallback mechanisms to ensure reliable data retrieval even when search operations encounter issues. All processing happens via FastAPI with Masumi payment integration for automated Cardano payments.

---

## 3. Key Capabilities

- **FRED Data Retrieval:** Automatic search and retrieval of economic time series with fallback to direct series ID access
- **Statistical Analysis:** Calculates MoM/YoY changes, percentile rankings, standard deviations, and historical context
- **Expert Interpretation:** Structured analysis with executive summaries, detailed data tables, implications, and FRED links

---

## 4. How It Works

**Input:** Natural language economic query via API (e.g., "unemployment rate in usa")

**Processing:**

- FRED Data Analyst searches FRED database, retrieves complete historical data, calculates 15+ statistical metrics
- Economic Advisor structures analysis into six sections: Introduction, Executive Summary, Detailed Analysis, Historical Context, Implications, Further Exploration

**Output:** Markdown-formatted analysis with data tables, calculated metrics, historical context, and expert interpretation

---

## 5. Transparency & Data Handling

|         Field         |                                     Details                                     |
| :-------------------: | :-----------------------------------------------------------------------------: |
|  Processing Location  |                         Server-side (user's deployment)                         |
|       LLMs Used       |                          OpenAI (default: gpt-5-nano)                           |
|   Third-Party Tools   |                  FRED API, Masumi Payment Service, OpenAI API                   |
|      Data Usage       | Queries sent to OpenAI (may be used for training). Economic data from FRED API. |
|    Data Retention     |           Queries NOT SAVED. Logs kept 12 months (no personal data).            |
| Data Storage Location |                        In-memory during processing only                         |
|   Security Measures   |        HTTPS encryption, environment variable API keys, no query storage        |
|        Privacy        |           Queries not saved. Sent to OpenAI per their privacy policy.           |
|      Legal Basis      |      Service "as is". FRED data source. AI analysis for information only.       |
|      User Rights      |            Access results via API. Export data. Queries not stored.             |
|      Access Logs      |                      12 months (job IDs, timestamps only)                       |
|    Output Formats     |                      JSON with markdown-formatted analysis                      |

---

## 6. Real-World Impact

- **Time Savings:** Reduces hours of manual work to 30-60 seconds
- **Accessibility:** Makes professional economic analysis accessible to non-economists
- **Reliability:** Automatic fallback mechanisms ensure data delivery even during API issues
- **Comprehensive:** Complete coverage of multi-part queries with all requested data
- **Decision Support:** Structured insights with economic, policy, market, and business implications

---

## 7. Who It's For

- **Enterprises:** Financial institutions, consulting firms, corporations needing rapid economic data analysis
- **Professionals:** Economists, financial analysts, researchers, journalists, policy advisors
- **Consumers:** Individual investors, students, general users seeking economic insights
- **General Users:** Anyone needing quick, professional FRED data analysis

---

## 8. How to Use the Agent

**Input Examples**

|        Input Field        |                  Example                   |
| :-----------------------: | :----------------------------------------: |
| identifier_from_purchaser |          "example_purchaser_123"           |
|      input_data.text      |         "unemployment rate in usa"         |
|      input_data.text      |   "What is the current inflation rate?"    |
|      input_data.text      | "Compare unemployment, inflation, and GDP" |

**Prompt Tips**

- Be specific: Include indicator names and time periods
- Multi-part queries supported: Ask for multiple indicators in one query
- Scope: FRED economic data only (not weather, recipes, etc.)
- Processing: 30-60 seconds average, payment required before execution
- Output: Complete analysis with data tables, metrics, and FRED links
