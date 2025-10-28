#!/usr/bin/env python3
"""
Quick CLI test for FREDagent with custom prompts
Usage: python test_custom.py "Your query here"
"""

import sys
from crew_definition import FREDEconomicCrew

def main():
    if len(sys.argv) < 2:
        print("\n" + "="*80)
        print("FRED Economic Data Agent - Custom Query Tester")
        print("="*80)
        print("\nUsage: python test_custom.py 'Your economic data query'")
        print("\nExamples:")
        print('  python test_custom.py "What is the current unemployment rate?"')
        print('  python test_custom.py "Show me GDP growth over the last 2 years"')
        print('  python test_custom.py "What is the federal funds rate?"')
        print("\n" + "="*80 + "\n")
        sys.exit(1)
    
    # Get the query from command line arguments
    query = ' '.join(sys.argv[1:])
    
    print("\n" + "="*80)
    print("🧪 FRED Economic Data Agent - Testing")
    print("="*80)
    print(f"\n📝 Query: {query}")
    print("\n" + "="*80)
    print("\n🚀 Running agent...\n")
    
    try:
        # Initialize and run the crew
        crew = FREDEconomicCrew()
        result = crew.crew.kickoff({"text": query})
        
        print("\n" + "="*80)
        print("📊 FRED AGENT RESPONSE")
        print("="*80)
        print(result)
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

