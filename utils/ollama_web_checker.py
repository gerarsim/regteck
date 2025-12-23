# utils/ollama_web_checker.py

import requests
import json
from typing import Dict, List, Any
from bs4 import BeautifulSoup
import re

class OllamaWebChecker:
    """
    Enables Ollama to check official compliance websites
    Implements function calling for real-time verification
    """

    def __init__(self, ollama_host: str = "http://localhost:11434"):
        self.ollama_host = ollama_host

        # Official sources to check
        self.official_sources = {
            'gdpr': {
                'url': 'https://gdpr.eu/',
                'api': 'https://gdpr.eu/search/',
                'name': 'GDPR Official'
            },
            'cssf': {
                'url': 'https://www.cssf.lu/en/',
                'api': 'https://www.cssf.lu/en/search/',
                'name': 'CSSF Luxembourg'
            },
            'eu_sanctions': {
                'url': 'https://www.sanctionsmap.eu/',
                'api': 'https://webgate.ec.europa.eu/fsd/fsf/public/files/xmlFullSanctionsList/',
                'name': 'EU Sanctions Map'
            },
            'ofac': {
                'url': 'https://sanctionssearch.ofac.treas.gov/',
                'api': 'https://www.treasury.gov/ofac/downloads/sdnlist.txt',
                'name': 'US OFAC'
            },
            'eba': {
                'url': 'https://www.eba.europa.eu/',
                'name': 'European Banking Authority'
            }
        }

        # Tools available to Ollama
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "check_gdpr_compliance",
                    "description": "Check GDPR official website for compliance requirements",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "topic": {
                                "type": "string",
                                "description": "GDPR topic to check (e.g., 'consent', 'data processing', 'right to erasure')"
                            }
                        },
                        "required": ["topic"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_cssf_regulations",
                    "description": "Check CSSF Luxembourg regulations and circulars",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "regulation_type": {
                                "type": "string",
                                "description": "Type of regulation (e.g., 'AML', 'banking', 'investment funds')"
                            }
                        },
                        "required": ["regulation_type"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "check_sanctions_list",
                    "description": "Check EU/OFAC sanctions lists for entities or individuals",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "entity_name": {
                                "type": "string",
                                "description": "Name of person or organization to check"
                            },
                            "list_type": {
                                "type": "string",
                                "enum": ["EU", "OFAC", "both"],
                                "description": "Which sanctions list to check"
                            }
                        },
                        "required": ["entity_name"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "get_latest_regulatory_update",
                    "description": "Get latest regulatory updates from official sources",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "source": {
                                "type": "string",
                                "enum": ["CSSF", "EBA", "ECB", "GDPR"],
                                "description": "Regulatory source"
                            },
                            "topic": {
                                "type": "string",
                                "description": "Topic of interest"
                            }
                        },
                        "required": ["source"]
                    }
                }
            }
        ]

    def check_gdpr_compliance(self, topic: str) -> Dict[str, Any]:
        """Check GDPR official website"""
        try:
            # Search GDPR.eu
            search_url = f"https://gdpr.eu/?s={topic.replace(' ', '+')}"
            response = requests.get(search_url, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract relevant articles
                articles = []
                for result in soup.find_all('article', limit=3):
                    title = result.find('h2')
                    content = result.find('p')
                    link = result.find('a', href=True)

                    if title and content:
                        articles.append({
                            'title': title.text.strip(),
                            'summary': content.text.strip()[:200],
                            'url': link['href'] if link else search_url
                        })

                return {
                    'source': 'GDPR Official',
                    'topic': topic,
                    'found': len(articles) > 0,
                    'articles': articles,
                    'last_checked': 'real-time',
                    'url': search_url
                }

        except Exception as e:
            return {
                'source': 'GDPR Official',
                'topic': topic,
                'error': str(e),
                'found': False
            }

    def check_cssf_regulations(self, regulation_type: str) -> Dict[str, Any]:
        """Check CSSF Luxembourg website"""
        try:
            # CSSF circulars and regulations
            base_url = "https://www.cssf.lu/en/"

            # Map regulation types to CSSF sections
            section_map = {
                'aml': 'supervision/aml-cft/',
                'banking': 'supervision/banking-supervision/',
                'investment': 'supervision/ucis-supervision/',
                'funds': 'supervision/ucis-supervision/'
            }

            section = section_map.get(regulation_type.lower(), 'supervision/')
            url = base_url + section

            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract recent circulars
                circulars = []
                for item in soup.find_all('div', class_='circular', limit=5):
                    title = item.find('h3') or item.find('h2')
                    date = item.find('time') or item.find('span', class_='date')

                    if title:
                        circulars.append({
                            'title': title.text.strip(),
                            'date': date.text.strip() if date else 'N/A',
                            'type': regulation_type
                        })

                return {
                    'source': 'CSSF Luxembourg',
                    'regulation_type': regulation_type,
                    'found': len(circulars) > 0,
                    'circulars': circulars,
                    'last_checked': 'real-time',
                    'url': url
                }

        except Exception as e:
            return {
                'source': 'CSSF Luxembourg',
                'regulation_type': regulation_type,
                'error': str(e),
                'found': False
            }

    def check_sanctions_list(self, entity_name: str, list_type: str = "both") -> Dict[str, Any]:
        """Check sanctions lists"""
        results = {
            'entity_name': entity_name,
            'found_on_sanctions_list': False,
            'lists_checked': [],
            'matches': []
        }

        # Check EU sanctions
        if list_type in ["EU", "both"]:
            try:
                # EU consolidated list
                eu_url = "https://webgate.ec.europa.eu/fsd/fsf/public/files/xmlFullSanctionsList/content?token=dG9rZW4tMjAxNw"
                response = requests.get(eu_url, timeout=15)

                if response.status_code == 200:
                    # Parse XML response
                    content = response.text.lower()
                    if entity_name.lower() in content:
                        results['found_on_sanctions_list'] = True
                        results['matches'].append({
                            'list': 'EU Consolidated',
                            'confidence': 'High',
                            'action': 'IMMEDIATE REJECTION REQUIRED'
                        })

                results['lists_checked'].append('EU')

            except Exception as e:
                results['lists_checked'].append(f'EU (error: {str(e)})')

        # Check OFAC
        if list_type in ["OFAC", "both"]:
            try:
                # OFAC SDN list
                ofac_url = "https://www.treasury.gov/ofac/downloads/sdnlist.txt"
                response = requests.get(ofac_url, timeout=15)

                if response.status_code == 200:
                    content = response.text.lower()
                    if entity_name.lower() in content:
                        results['found_on_sanctions_list'] = True
                        results['matches'].append({
                            'list': 'US OFAC SDN',
                            'confidence': 'High',
                            'action': 'IMMEDIATE REJECTION REQUIRED'
                        })

                results['lists_checked'].append('OFAC')

            except Exception as e:
                results['lists_checked'].append(f'OFAC (error: {str(e)})')

        results['last_checked'] = 'real-time'
        results['risk_level'] = 'EXTREME' if results['found_on_sanctions_list'] else 'NONE'

        return results

    def get_latest_regulatory_update(self, source: str, topic: str = None) -> Dict[str, Any]:
        """Get latest regulatory updates"""

        source_urls = {
            'CSSF': 'https://www.cssf.lu/en/news-publications/',
            'EBA': 'https://www.eba.europa.eu/news-press',
            'ECB': 'https://www.ecb.europa.eu/press/html/index.en.html',
            'GDPR': 'https://gdpr.eu/news/'
        }

        try:
            url = source_urls.get(source)
            if not url:
                return {'error': f'Unknown source: {source}'}

            response = requests.get(url, timeout=10)

            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')

                # Extract recent news/updates
                updates = []
                for item in soup.find_all(['article', 'div'], class_=re.compile('news|article|update'), limit=5):
                    title = item.find(['h2', 'h3', 'h4'])
                    date = item.find('time') or item.find('span', class_=re.compile('date'))

                    if title:
                        text = title.text.strip()
                        if not topic or topic.lower() in text.lower():
                            updates.append({
                                'title': text,
                                'date': date.text.strip() if date else 'Recent',
                                'source': source
                            })

                return {
                    'source': source,
                    'topic': topic,
                    'updates': updates,
                    'count': len(updates),
                    'last_checked': 'real-time',
                    'url': url
                }

        except Exception as e:
            return {
                'source': source,
                'topic': topic,
                'error': str(e),
                'updates': []
            }

    def analyze_with_web_checking(self, text: str, model: str = "llama3.2:3b") -> Dict[str, Any]:
        """
        Analyze document with Ollama + real-time web checking
        """

        # Step 1: Initial analysis with function calling enabled
        prompt = f"""You are a Luxembourg compliance expert. Analyze this document for regulatory compliance.

For any compliance question where you need current information, use the available functions to check official sources:
- check_gdpr_compliance() for GDPR questions
- check_cssf_regulations() for Luxembourg banking regulations  
- check_sanctions_list() for sanctions screening
- get_latest_regulatory_update() for recent changes

Document to analyze:
{text}

Provide a comprehensive compliance analysis."""

        # Call Ollama with tools
        response = requests.post(
            f"{self.ollama_host}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "tools": self.tools,
                "stream": False
            }
        )

        if response.status_code != 200:
            return {"error": f"Ollama request failed: {response.status_code}"}

        result = response.json()

        # Step 2: Process tool calls
        message = result.get('message', {})
        tool_calls = message.get('tool_calls', [])

        web_checks_performed = []

        for tool_call in tool_calls:
            function_name = tool_call['function']['name']
            arguments = tool_call['function']['arguments']

            # Execute the function
            if function_name == 'check_gdpr_compliance':
                check_result = self.check_gdpr_compliance(**arguments)
                web_checks_performed.append(check_result)

            elif function_name == 'check_cssf_regulations':
                check_result = self.check_cssf_regulations(**arguments)
                web_checks_performed.append(check_result)

            elif function_name == 'check_sanctions_list':
                check_result = self.check_sanctions_list(**arguments)
                web_checks_performed.append(check_result)

            elif function_name == 'get_latest_regulatory_update':
                check_result = self.get_latest_regulatory_update(**arguments)
                web_checks_performed.append(check_result)

        # Step 3: Re-analyze with web results
        if web_checks_performed:
            followup_prompt = f"""Based on the official sources checked:

{json.dumps(web_checks_performed, indent=2)}

Update your compliance analysis with this real-time verified information."""

            followup_response = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": model,
                    "messages": [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": message.get('content', '')},
                        {"role": "user", "content": followup_prompt}
                    ],
                    "stream": False
                }
            )

            if followup_response.status_code == 200:
                final_result = followup_response.json()
                return {
                    'analysis': final_result.get('message', {}).get('content', ''),
                    'web_checks': web_checks_performed,
                    'sources_verified': len(web_checks_performed),
                    'real_time_verification': True
                }

        return {
            'analysis': message.get('content', ''),
            'web_checks': web_checks_performed,
            'sources_verified': len(web_checks_performed),
            'real_time_verification': bool(web_checks_performed)
        }


# Example usage
if __name__ == "__main__":
    checker = OllamaWebChecker()

    # Test GDPR check
    result = checker.check_gdpr_compliance("right to be forgotten")
    print("GDPR Check:", json.dumps(result, indent=2))

    # Test sanctions check
    sanctions = checker.check_sanctions_list("Vladimir Putin", "both")
    print("\nSanctions Check:", json.dumps(sanctions, indent=2))

    # Test full analysis
    test_doc = """
    We process personal data of EU citizens without explicit consent.
    Our banking services are available to clients in Iran and North Korea.
    """

    analysis = checker.analyze_with_web_checking(test_doc)
    print("\nFull Analysis:", json.dumps(analysis, indent=2))🎉