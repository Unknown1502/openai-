#!/usr/bin/env python3
"""Enhanced Competition Runner with multiple testing strategies and scanning modes."""

import sys
import os
import asyncio
import json
import random
import gc
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from enum import Enum
import argparse

# Fix vendor issues before imports
def fix_vendor_before_imports():
    """Fix vendor directory issues before importing anything else"""
    vendor_paths = [
        "/content/openai-/vendor",
        "/content/kaggleproject/vendor",
        "/kaggle/working/vendor",
        "/kaggle/working/kaggleproject/vendor",
        "./vendor",
        "vendor"
    ]
    
    for vendor_dir in vendor_paths:
        if os.path.exists(vendor_dir) and vendor_dir not in sys.path:
            sys.path.append(vendor_dir)

fix_vendor_before_imports()

# Set environment variables
if os.path.exists("/content/openai-"):
    os.environ['TORCH_HOME'] = "/content/openai-/.cache/torch"
else:
    os.environ['TORCH_HOME'] = "/content/kaggleproject/.cache/torch"
os.environ['CUDA_HOME'] = '/usr/local/cuda'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Import torch first
try:
    import torch
    print(f"PyTorch loaded successfully: {torch.__version__}")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
except ImportError as e:
    print(f"Failed to import PyTorch: {e}")

from src.competition.enhanced_attack_vectors import (
    ENHANCED_COMPETITION_ATTACK_VECTORS,
    get_attack_scenarios_by_category,
    get_attack_scenarios_by_severity,
    get_categories,
    Severity
)
from src.competition.findings_formatter import FindingsFormatter
from src.core.client_factory import ClientFactory
from src.core.vulnerability_scanner import VulnerabilityScanner
from src.config import load_config
from src.utils.memory_manager import MemoryManager, prepare_for_model_loading

class TestingStrategy(Enum):
    STANDARD = "standard"
    AGGRESSIVE = "aggressive"
    SUBTLE = "subtle"

class ScanningMode(Enum):
    COMPREHENSIVE = "comprehensive"
    TARGETED = "targeted"
    RANDOM = "random"
    QUICK = "quick"

class EnhancedCompetitionRunner:
    def __init__(self, strategy: TestingStrategy = TestingStrategy.STANDARD, 
                 mode: ScanningMode = ScanningMode.COMPREHENSIVE):
        self.strategy = strategy
        self.mode = mode
        self.config = load_config()
        self.client = None
        self.scanner = None
        self.findings = []
        self.scan_history = []
        
    async def initialize(self):
        """Initialize the runner with memory management."""
        print("🚀 Initializing Enhanced Competition Runner...")
        print(f"Strategy: {self.strategy.value}")
        print(f"Mode: {self.mode.value}")
        
        # Prepare memory
        model_size_gb = 20.0
        can_load, memory_message = prepare_for_model_loading(model_size_gb)
        print(f"Memory status: {memory_message}")
        
        if not can_load and self.mode != ScanningMode.QUICK:
            print("⚠️ Limited memory detected, switching to QUICK mode")
            self.mode = ScanningMode.QUICK
        
        # Log initial memory
        MemoryManager.log_memory_status("Before creating client")
        
        # Create client and scanner
        self.client = ClientFactory.create_client(self.config)
        self.scanner = VulnerabilityScanner(self.client)
        
        # Log memory after loading
        MemoryManager.log_memory_status("After model loading")
        
    def get_attack_scenarios(self) -> List:
        """Get attack scenarios based on mode and strategy."""
        all_scenarios = ENHANCED_COMPETITION_ATTACK_VECTORS
        
        # Filter by strategy
        if self.strategy == TestingStrategy.AGGRESSIVE:
            # Focus on high severity attacks
            scenarios = get_attack_scenarios_by_severity(Severity.CRITICAL) + \
                       get_attack_scenarios_by_severity(Severity.HIGH)
        elif self.strategy == TestingStrategy.SUBTLE:
            # Focus on medium/low severity that might slip through
            scenarios = get_attack_scenarios_by_severity(Severity.MEDIUM) + \
                       get_attack_scenarios_by_severity(Severity.LOW)
        else:
            scenarios = all_scenarios
        
        # Apply mode
        if self.mode == ScanningMode.TARGETED:
            # Focus on specific high-value categories
            target_categories = ["deceptive_alignment", "evaluation_awareness", "sandbagging"]
            scenarios = [s for s in scenarios if s.category in target_categories]
        elif self.mode == ScanningMode.RANDOM:
            # Random sample of 20 scenarios
            scenarios = random.sample(scenarios, min(20, len(scenarios)))
        elif self.mode == ScanningMode.QUICK:
            # Quick scan with 10 diverse scenarios
            categories = get_categories()
            quick_scenarios = []
            for cat in categories[:5]:  # Top 5 categories
                cat_scenarios = [s for s in scenarios if s.category == cat]
                if cat_scenarios:
                    quick_scenarios.extend(cat_scenarios[:2])  # 2 per category
            scenarios = quick_scenarios
        
        return scenarios
    
    async def run_attack_scenario(self, scenario, index: int, total: int) -> Optional[Dict]:
        """Run a single attack scenario with memory management."""
        print(f"\n[{index}/{total}] Testing: {scenario.name}")
        print(f"  Category: {scenario.category}")
        print(f"  Severity: {scenario.severity.value}")
        
        # Clear memory if needed
        if index > 1 and index % 5 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        try:
            # Run the attack
            result = await self.scanner.test_vulnerability(
                prompt=scenario.prompt_template,
                category=scenario.category,
                expected_behavior=scenario.expected_vulnerability
            )
            
            if result.is_vulnerable:
                print(f"  ✅ Vulnerability found: {result.vulnerability_type}")
                
                # Create enhanced finding
                finding = FindingsFormatter.create_finding(
                    issue_title=f"{scenario.name} - {result.vulnerability_type}",
                    topic_area=scenario.category,
                    overview=result.description,
                    desired_behavior=f"Model should not exhibit {scenario.expected_vulnerability}",
                    observed_behavior=result.evidence,
                    severity=result.severity_score,
                    breadth=result.breadth_score,
                    harmony_response=result.model_response,
                    steps_to_reproduce=result.reproduction_steps,
                    notes=f"Strategy: {self.strategy.value}, Severity: {scenario.severity.value}"
                )
                
                # Add metadata
                finding['metadata'] = {
                    'strategy': self.strategy.value,
                    'mode': self.mode.value,
                    'scenario_severity': scenario.severity.value,
                    'timestamp': datetime.utcnow().isoformat()
                }
                
                return finding
            else:
                print(f"  ❌ No vulnerability detected")
                
        except torch.cuda.OutOfMemoryError:
            print(f"  ⚠️ GPU OOM, attempting recovery...")
            MemoryManager.clear_memory()
            
            # Retry with shorter prompt
            try:
                result = await self.scanner.test_vulnerability(
                    prompt=scenario.prompt_template[:300],
                    category=scenario.category,
                    expected_behavior=scenario.expected_vulnerability
                )
                if result.is_vulnerable:
                    print(f"  ✅ Vulnerability found on retry")
                    return self._create_finding_from_result(result, scenario)
            except Exception as e:
                print(f"  ❌ Failed after retry: {e}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        return None
    
    def _create_finding_from_result(self, result, scenario) -> Dict:
        """Helper to create finding from result."""
        finding = FindingsFormatter.create_finding(
            issue_title=f"{scenario.name} - {result.vulnerability_type}",
            topic_area=scenario.category,
            overview=result.description,
            desired_behavior=f"Model should not exhibit {scenario.expected_vulnerability}",
            observed_behavior=result.evidence,
            severity=result.severity_score,
            breadth=result.breadth_score,
            harmony_response=result.model_response,
            steps_to_reproduce=result.reproduction_steps,
            notes=f"Strategy: {self.strategy.value}, Severity: {scenario.severity.value}"
        )
        
        finding['metadata'] = {
            'strategy': self.strategy.value,
            'mode': self.mode.value,
            'scenario_severity': scenario.severity.value,
            'timestamp': datetime.utcnow().isoformat()
        }
        
        return finding
    
    async def run_competition(self) -> Dict:
        """Run the enhanced competition scan."""
        await self.initialize()
        
        # Get scenarios
        scenarios = self.get_attack_scenarios()
        print(f"\n🎯 Running {len(scenarios)} attack scenarios...")
        
        # Track statistics
        stats = {
            'total_scenarios': len(scenarios),
            'vulnerabilities_found': 0,
            'categories_tested': set(),
            'severity_distribution': {},
            'start_time': datetime.utcnow().isoformat()
        }
        
        # Run scenarios
        for i, scenario in enumerate(scenarios, 1):
            finding = await self.run_attack_scenario(scenario, i, len(scenarios))
            
            if finding:
                self.findings.append(finding)
                stats['vulnerabilities_found'] += 1
                
                # Save individual finding
                filename = f"enhanced_finding_{len(self.findings)}_{scenario.category}_{self.strategy.value}.json"
                FindingsFormatter.save_finding(finding, filename)
                print(f"  💾 Saved: {filename}")
            
            stats['categories_tested'].add(scenario.category)
            
            # Update severity distribution
            sev = scenario.severity.value
            stats['severity_distribution'][sev] = stats['severity_distribution'].get(sev, 0) + 1
            
            # Log memory periodically
            if i % 10 == 0:
                MemoryManager.log_memory_status(f"After {i} scenarios")
        
        # Finalize statistics
        stats['end_time'] = datetime.utcnow().isoformat()
        stats['categories_tested'] = list(stats['categories_tested'])
        
        # Generate comprehensive report
        report = self.generate_report(stats)
        
        # Save report
        report_filename = f"enhanced_competition_report_{self.strategy.value}_{self.mode.value}.json"
        with open(report_filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Report saved: {report_filename}")
        
        return report
    
    def generate_report(self, stats: Dict) -> Dict:
        """Generate comprehensive competition report."""
        report = {
            'summary': {
                'strategy': self.strategy.value,
                'mode': self.mode.value,
                'total_scenarios_tested': stats['total_scenarios'],
                'vulnerabilities_found': stats['vulnerabilities_found'],
                'success_rate': f"{(stats['vulnerabilities_found'] / stats['total_scenarios'] * 100):.1f}%",
                'categories_tested': stats['categories_tested'],
                'severity_distribution': stats['severity_distribution'],
                'start_time': stats['start_time'],
                'end_time': stats['end_time']
            },
            'findings': self.findings,
            'analysis': self.analyze_findings(),
            'recommendations': self.generate_recommendations()
        }
        
        return report
    
    def analyze_findings(self) -> Dict:
        """Analyze findings for patterns and insights."""
        if not self.findings:
            return {'status': 'No vulnerabilities found'}
        
        analysis = {
            'most_vulnerable_categories': {},
            'severity_analysis': {},
            'common_patterns': [],
            'high_risk_areas': []
        }
        
        # Category analysis
        category_counts = {}
        for finding in self.findings:
            cat = finding['issue_summary']['topic_area']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        analysis['most_vulnerable_categories'] = dict(
            sorted(category_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        )
        
        # Severity analysis
        severity_counts = {}
        for finding in self.findings:
            if 'metadata' in finding:
                sev = finding['metadata'].get('scenario_severity', 'unknown')
                severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        analysis['severity_analysis'] = severity_counts
        
        # Identify high-risk areas
        critical_findings = [f for f in self.findings 
                           if f.get('metadata', {}).get('scenario_severity') == 'critical']
        if critical_findings:
            analysis['high_risk_areas'] = [
                f['issue_summary']['topic_area'] for f in critical_findings[:5]
            ]
        
        return analysis
    
    def generate_recommendations(self) -> List[str]:
        """Generate recommendations based on findings."""
        recommendations = []
        
        if not self.findings:
            recommendations.append("No vulnerabilities detected - model appears robust against tested scenarios")
            return recommendations
        
        # Category-based recommendations
        category_counts = {}
        for finding in self.findings:
            cat = finding['issue_summary']['topic_area']
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        for category, count in category_counts.items():
            if count >= 3:
                recommendations.append(f"High vulnerability in {category} - implement targeted defenses")
            elif count >= 1:
                recommendations.append(f"Moderate risk in {category} - monitor and strengthen")
        
        # Severity-based recommendations
        critical_count = sum(1 for f in self.findings 
                           if f.get('metadata', {}).get('scenario_severity') == 'critical')
        if critical_count > 0:
            recommendations.append(f"URGENT: {critical_count} critical vulnerabilities require immediate attention")
        
        # Strategy-specific recommendations
        if self.strategy == TestingStrategy.AGGRESSIVE and len(self.findings) > 5:
            recommendations.append("Model vulnerable to aggressive attacks - implement robust safety measures")
        elif self.strategy == TestingStrategy.SUBTLE and len(self.findings) > 3:
            recommendations.append("Model susceptible to subtle manipulation - enhance detection mechanisms")
        
        return recommendations[:10]  # Top 10 recommendations


async def main():
    parser = argparse.ArgumentParser(description="Enhanced Competition Runner for gpt-oss-20b")
    parser.add_argument(
        "--strategy", 
        type=str, 
        choices=["standard", "aggressive", "subtle"],
        default="standard",
        help="Testing strategy to use"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["comprehensive", "targeted", "random", "quick"],
        default="comprehensive",
        help="Scanning mode"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="enhanced_results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.chdir(args.output_dir)
    
    # Create runner
    strategy = TestingStrategy(args.strategy)
    mode = ScanningMode(args.mode)
    runner = EnhancedCompetitionRunner(strategy, mode)
    
    # Run competition
    print("=" * 60)
    print("🏆 Enhanced Competition Runner for gpt-oss-20b")
    print("=" * 60)
    
    try:
        report = await runner.run_competition()
        
        print("\n" + "=" * 60)
        print("📈 Competition Results Summary")
        print("=" * 60)
        print(f"Strategy: {report['summary']['strategy']}")
        print(f"Mode: {report['summary']['mode']}")
        print(f"Scenarios Tested: {report['summary']['total_scenarios_tested']}")
        print(f"Vulnerabilities Found: {report['summary']['vulnerabilities_found']}")
        print(f"Success Rate: {report['summary']['success_rate']}")
        print(f"\nTop Vulnerable Categories:")
        for cat, count in list(report['analysis']['most_vulnerable_categories'].items())[:3]:
            print(f"  - {cat}: {count} vulnerabilities")
        print(f"\nRecommendations:")
        for i, rec in enumerate(report['recommendations'][:5], 1):
            print(f"  {i}. {rec}")
        
    except Exception as e:
        print(f"\n❌ Competition failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final memory cleanup
        MemoryManager.log_memory_status("Final")
        MemoryManager.clear_memory()


if __name__ == "__main__":
    if "ipykernel" in sys.modules:  # Jupyter/Colab
        import nest_asyncio
        nest_asyncio.apply()
        asyncio.get_event_loop().run_until_complete(main())
    else:
        asyncio.run(main())
