from crewai import Agent, Crew, Task
from crewai import LLM
from testing.logging_config import get_logger
from crewai.tools import tool
from fredapi import Fred
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np

load_dotenv()

@tool("FRED Search Tool")
def fred_search_tool(query: str) -> str:
    """
    Search the FRED database for economic data series matching the query.
    Returns series IDs, titles, and descriptions of matching datasets.
    """
    try:
        fred_api_key = os.getenv("FRED_API_KEY")
        if not fred_api_key:
            return "Error: FRED_API_KEY not found in environment variables. Please add it to your .env file."
        
        fred = Fred(api_key=fred_api_key)
        results = fred.search(query, limit=10)
        
        if results.empty:
            return f"No results found for query: '{query}'"
        
        output = f"Found {len(results)} series matching '{query}':\n\n"
        for idx, (series_id, row) in enumerate(results.iterrows(), 1):
            output += f"{idx}. {row.get('title', 'N/A')} (ID: {series_id})\n"
            output += f"   Description: {row.get('notes', 'No description available')[:200]}...\n"
            output += f"   Frequency: {row.get('frequency_short', 'N/A')} | Units: {row.get('units_short', 'N/A')}\n\n"
        
        return output
    except Exception as e:
        return f"Error searching FRED: {str(e)}"

@tool("FRED Data Retrieval Tool")
def fred_data_tool(series_id: str) -> str:
    """
    Retrieve actual economic data from FRED for a specific series ID with comprehensive analysis.
    Returns recent data points, calculated metrics (MoM, YoY, percentiles), and statistical context.
    """
    try:
        fred_api_key = os.getenv("FRED_API_KEY")
        if not fred_api_key:
            return "Error: FRED_API_KEY not found in environment variables."
        
        fred = Fred(api_key=fred_api_key)
        
        # Get series info
        info = fred.get_series_info(series_id)
        
        # Get complete data history
        data = fred.get_series(series_id)
        
        if data.empty:
            return f"No data available for series ID: {series_id}"
        
        # Get recent data points (last 24 for comprehensive analysis - 2 years monthly or 6 years quarterly)
        recent_data = data.tail(24)
        
        # Calculate metrics
        current_value = data.iloc[-1]
        
        # MoM change (if monthly or higher frequency)
        mom_change = None
        mom_pct = None
        if len(data) >= 2:
            prev_value = data.iloc[-2]
            mom_change = current_value - prev_value
            if prev_value != 0:
                mom_pct = (mom_change / prev_value) * 100
        
        # YoY change (if we have 12+ months of data)
        yoy_change = None
        yoy_pct = None
        freq = info.get('frequency_short', 'N/A')
        lookback = 12 if freq in ['M', 'Monthly'] else 4 if freq in ['Q', 'Quarterly'] else 1
        if len(data) >= lookback + 1:
            year_ago_value = data.iloc[-(lookback + 1)]
            yoy_change = current_value - year_ago_value
            if year_ago_value != 0:
                yoy_pct = (yoy_change / year_ago_value) * 100
        
        # Historical statistics
        mean_value = data.mean()
        std_value = data.std()
        min_value = data.min()
        max_value = data.max()
        
        # Percentile rank of current value
        percentile = (data < current_value).sum() / len(data) * 100
        
        # Standard deviations from mean
        std_from_mean = (current_value - mean_value) / std_value if std_value != 0 else 0
        
        # 3-month or 3-period average
        period_avg = data.tail(3).mean() if len(data) >= 3 else current_value
        
        # Build comprehensive output with FIXED f-string syntax
        output = f"=== SERIES ANALYSIS: {info.get('title', series_id)} ===\n\n"
        output += f"📊 CURRENT DATA:\n"
        output += f"Series ID: {series_id}\n"
        # FIXED: Proper f-string syntax
        output += f"Current Value: {'N/A' if pd.isna(current_value) else f'{current_value:.2f}'}\n"
        output += f"Date: {data.index[-1].strftime('%Y-%m-%d')}\n"
        output += f"Frequency: {info.get('frequency', 'N/A')}\n"
        output += f"Units: {info.get('units', 'N/A')}\n"
        output += f"Seasonal Adjustment: {info.get('seasonal_adjustment', 'N/A')}\n\n"
        
        output += f"📈 CALCULATED METRICS:\n"
        if mom_change is not None:
            output += f"Month-over-Month Change: {mom_change:+.2f} ({mom_pct:+.2f}%)\n"
        if yoy_change is not None:
            output += f"Year-over-Year Change: {yoy_change:+.2f} ({yoy_pct:+.2f}%)\n"
        output += f"3-Period Average: {period_avg:.2f}\n\n"
        
        output += f"📉 HISTORICAL CONTEXT:\n"
        output += f"Historical Mean: {mean_value:.2f}\n"
        output += f"Standard Deviation: {std_value:.2f}\n"
        output += f"Historical Range: {min_value:.2f} to {max_value:.2f}\n"
        output += f"Current Percentile Rank: {percentile:.1f}th percentile\n"
        output += f"Distance from Mean: {std_from_mean:+.2f} standard deviations\n"
        output += f"Data Range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}\n"
        output += f"Total Observations: {len(data)}\n\n"
        
        output += f"📋 RECENT DATA POINTS (Last 15):\n"
        for date, value in recent_data.tail(15).items():
            output += f"  {date.strftime('%Y-%m-%d')}: {value:.2f}\n"
        
        # Add summary of full dataset
        output += f"\n📊 FULL DATASET SUMMARY:\n"
        output += f"Total data points retrieved: {len(data)}\n"
        output += f"Oldest data: {data.index[0].strftime('%Y-%m-%d')} = {data.iloc[0]:.2f}\n"
        output += f"Newest data: {data.index[-1].strftime('%Y-%m-%d')} = {data.iloc[-1]:.2f}\n"
        output += f"Average over entire period: {mean_value:.2f}\n"
        output += f"Peak value: {max_value:.2f} on {data.idxmax().strftime('%Y-%m-%d')}\n"
        output += f"Trough value: {min_value:.2f} on {data.idxmin().strftime('%Y-%m-%d')}\n"
        
        output += f"\n🔗 View on FRED: https://fred.stlouisfed.org/series/{series_id}\n"
        
        return output
    except Exception as e:
        return f"Error retrieving data for {series_id}: {str(e)}"

@tool("FRED Series Info Tool")
def fred_series_info_tool(series_id: str) -> str:
    """
    Get detailed information about a FRED data series including metadata and source information.
    """
    try:
        fred_api_key = os.getenv("FRED_API_KEY")
        if not fred_api_key:
            return "Error: FRED_API_KEY not found in environment variables."
        
        fred = Fred(api_key=fred_api_key)
        info = fred.get_series_info(series_id)
        
        output = f"Series Information for {series_id}:\n\n"
        output += f"Title: {info.get('title', 'N/A')}\n"
        output += f"Observation Start: {info.get('observation_start', 'N/A')}\n"
        output += f"Observation End: {info.get('observation_end', 'N/A')}\n"
        output += f"Frequency: {info.get('frequency', 'N/A')}\n"
        output += f"Units: {info.get('units', 'N/A')}\n"
        output += f"Seasonal Adjustment: {info.get('seasonal_adjustment', 'N/A')}\n"
        output += f"Last Updated: {info.get('last_updated', 'N/A')}\n"
        output += f"Popularity: {info.get('popularity', 'N/A')}\n\n"
        output += f"Notes: {info.get('notes', 'No notes available')}\n\n"
        output += f"🔗 View on FRED: https://fred.stlouisfed.org/series/{series_id}\n"
        
        return output
    except Exception as e:
        return f"Error getting info for {series_id}: {str(e)}"


class FREDEconomicCrew:
    """
    A specialized CrewAI crew for querying and analyzing FRED economic data.
    Enhanced with analytical capabilities for comprehensive economic analysis.
    """
    def __init__(self, verbose=True, logger=None, model=None, temperature=None):
        self.verbose = verbose
        self.logger = logger or get_logger(__name__)
        # Configure LLM - support custom model and temperature, default to gpt-5-nano
        if model:
            llm_params = {"model": model}
            if temperature is not None:
                llm_params["temperature"] = temperature
            self.llm = LLM(**llm_params)
        else:
            self.llm = LLM(model="gpt-5-nano")
        self.crew = self.create_crew()
        self.logger.info("FRED Economic Crew initialized")

    def create_crew(self):
        self.logger.info("Creating FRED economic data crew")
        
        # Agent 1: FRED Data Analyst - Enhanced with analytical requirements
        fred_analyst = Agent(
            role='Senior FRED Data Analyst',
            goal='Retrieve ALL relevant economic data series requested and provide comprehensive statistical analysis',
            backstory="""You are a senior economic data analyst with deep expertise in the Federal Reserve 
            Economic Data (FRED) database. You have a PhD in Economics and 15 years of experience analyzing 
            economic indicators. You are meticulous about retrieving ALL data series mentioned in queries - 
            if someone asks for 3 metrics, you retrieve ALL 3, not just one. You understand economic terminology, 
            series IDs, and know how to find both current and historical data. You always use the data retrieval 
            tool to get actual numbers with calculated metrics, never just search results.
            
            CRITICAL GUARDRAILS:
            - ALWAYS call tools with single string parameters. Example: fred_search_tool("unemployment rate") NOT fred_search_tool(["query": "unemployment rate"])
            - NEVER pass JSON arrays or multiple parameters to tools - use single string arguments only
            - If a tool fails 2-3 times, try a different series ID or inform the user about the issue.
            - If FRED search returns NO results or empty data, you MUST immediately inform the user that 
              the requested data is not available in FRED. DO NOT make up data or provide generic responses.
            - You ONLY work with Federal Reserve Economic Data. If a query is clearly outside economics 
              (e.g., weather, recipes, entertainment), politely inform the user this is outside your scope.
            - Never hallucinate data. If you cannot retrieve actual data, say so explicitly.""",
            tools=[fred_search_tool, fred_data_tool, fred_series_info_tool],
            llm=self.llm,
            verbose=self.verbose
        )

        # Agent 2: Economic Advisor - Enhanced to provide structured, actionable analysis with comprehensive data
        economic_advisor = Agent(
            role='Chief Economic Interpreter',
            goal='Transform raw economic data into comprehensive, actionable insights with full data presentation and historical context',
            backstory="""You are the Chief Economist at a major financial institution with 20 years of experience 
            interpreting economic data for investors, policymakers, and business leaders. You are known for 
            providing COMPREHENSIVE analysis that includes both the raw data AND the insights. You always provide:
            
            1. INTRODUCTION with the user's original request and what data was analyzed
            2. EXECUTIVE SUMMARY with key metrics and changes
            3. DETAILED DATA ANALYSIS with complete metrics, recent data tables, and calculated changes
            4. HISTORICAL CONTEXT explaining if values are high/low vs historical norms with statistical significance
            5. INTERPRETATION of what the data means for the economy, policy, and markets
            6. ACTIONABLE GUIDANCE for different stakeholders
            7. FURTHER EXPLORATION links to related FRED series
            
            You format responses in clear sections with bullet points and comprehensive tables. You make 
            complex economics accessible while showing ALL the data that was retrieved. You NEVER give generic 
            boilerplate - every insight is specific to the query. When users ask for comparisons, you create 
            detailed comparison tables. When they ask about historical periods, you show that specific historical data.
            
            CRITICAL GUARDRAILS:
            - ALWAYS provide a final, complete answer. NEVER ask follow-up questions or leave the analysis incomplete.
            - Show ALL the raw data that was successfully retrieved, not just summaries
            - Exclude N/A values from output - only show calculated metrics that have real data
            - Only analyze data that was successfully retrieved. If data retrieval failed, acknowledge 
              this clearly and don't fabricate numbers.
            - If NO data was retrieved or all searches failed, you MUST inform the user that the requested 
              information is not available in FRED. Provide helpful suggestions for alternative queries.
            - Never provide analysis without actual data. Never hallucinate numbers.
            - If the query is outside the scope of FRED economic data, politely explain the agent's 
              limitations and what types of queries it can handle.
            - Always include the complete dataset statistics and recent data points in organized tables.
            - Make your best interpretation of the data and provide a complete analysis - don't ask if the user wants more.""",
            llm=self.llm,
            verbose=self.verbose
        )

        self.logger.info("Created enhanced FRED analyst and economic advisor agents")

        crew = Crew(
            agents=[fred_analyst, economic_advisor],
            tasks=[
                Task(
                    description="""Analyze this economic data query and retrieve ALL relevant data: {text}
                    
                    CRITICAL REQUIREMENTS:
                    1. Identify EVERY economic indicator mentioned in the query
                    2. If the query asks for multiple metrics (e.g., "compare A, B, and C"), retrieve ALL of them
                    3. If the query mentions specific time periods (e.g., "2008 crisis"), retrieve data from that period
                    4. Use fred_data_tool to get actual data with calculations - don't just search
                    5. Retrieve enough historical data to provide meaningful context
                    6. If a tool fails 2-3 times, try alternative series IDs or report the issue
                    
                    STEPS:
                    1. Use fred_search_tool with a single string query parameter. Example: fred_search_tool("unemployment rate")
                    2. Find the relevant series ID from the search results
                    3. Use fred_data_tool with a single string series_id parameter. Example: fred_data_tool("UNRATE")
                    4. NEVER pass JSON arrays or multiple parameters - ALWAYS use single string arguments
                    5. Verify you've retrieved data for EVERY part of the query
                    
                    EARLY EXIT CONDITIONS (Stop immediately and report):
                    - If FRED search returns NO results for the query
                    - If all data retrieval attempts fail
                    - If the query appears to be outside FRED's scope (non-economic)
                    - If no relevant economic indicators can be identified
                    
                    FORBIDDEN:
                    - Never provide search results without retrieving actual data
                    - Never answer only part of a multi-part question
                    - Never skip historical periods specifically mentioned
                    - Never retry the same failing tool more than 3 times
                    - Never fabricate data when retrieval fails
                    """,
                    expected_output="""Complete data retrieval including:
                    - Actual data values for ALL series mentioned in query
                    - Calculated metrics (MoM, YoY, percentiles) for each series
                    - Historical context data if requested
                    - All series metadata and FRED links
                    - Clear indication if any data retrieval failed
                    - If NO data found: explicit message stating data is unavailable with suggestions""",
                    agent=fred_analyst
                ),
                Task(
                    description="""Transform the retrieved FRED data into a comprehensive, actionable analysis.
                    
                    REQUIRED STRUCTURE (BE CONCISE):
                    
                    ## 🏠 INTRODUCTION
                    - User's original request: {text}
                    - Brief 1-2 sentence summary of what economic data was analyzed
                    - Data source and time period covered
                    
                    ## 📊 EXECUTIVE SUMMARY
                    - 3-5 bullet points with key findings
                    - Current values and most important changes
                    - One-line historical context (e.g., "highest since 2008")
                    
                    ## 📈 DETAILED DATA ANALYSIS
                    - Show current value, changes (MoM/YoY), and percentile rank
                    - Recent data table (last 10 periods) with VALUES ONLY (exclude N/A)
                    - Brief trend description
                    - Historical summary statistics (mean, min, max)
                    
                    ## 🔍 HISTORICAL CONTEXT
                    - Is current value high/low relative to historical norms?
                    - Statistical significance (standard deviations from mean)
                    
                    ## 💡 WHAT THIS MEANS
                    - Economic implications (1-2 sentences)
                    - Policy implications (1-2 sentences)
                    - Market implications (1-2 sentences)
                    
                    ## 🔗 FURTHER EXPLORATION
                    - Direct FRED link for the series
                    - 2-3 related indicators to explore
                    
                    CRITICAL REQUIREMENTS:
                    1. NEVER ask follow-up questions - provide a complete, final answer
                    2. Exclude N/A values from all output - only show calculated metrics with real data
                    3. Keep INTRODUCTION concise - 2-3 sentences max
                    4. Use specific numbers from actual data retrieved
                    5. Make your best interpretation and provide complete analysis
                    
                    FORBIDDEN:
                    - Never ask "Would you like me to..."
                    - Never say "If you'd like, I can..."
                    - Never include N/A in data tables or output
                    - Never leave analysis incomplete
                    - Never provide generic boilerplate""",
                    expected_output="""Concise economic analysis with:
                    - Introduction showing user's request and brief analysis overview
                    - Executive summary with 3-5 key findings
                    - Detailed data with current values, changes, and recent data table (NO N/A values)
                    - Historical context with statistical significance
                    - Brief implications for economy, policy, and markets
                    - FRED links for further exploration
                    - Complete final answer with no follow-up questions""",
                    agent=economic_advisor
                )
            ],
            verbose=True
        )
        
        self.logger.info("FRED Economic Crew setup completed with enhanced analytical capabilities")
        return crew