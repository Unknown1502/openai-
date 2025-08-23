#!/bin/bash

# ============================================================================
# GPT-OSS-20B Red Teaming Competition - Complete Execution Script
# Based on: The Red Teamer's Compendium Research
# ============================================================================

echo "========================================================================="
echo "🏆 GPT-OSS-20B Red Teaming Competition - Complete Runner"
echo "========================================================================="
echo "📚 Research-Based Attack Strategy Implementation"
echo "🎯 Target: GPT-OSS-20B Mixture-of-Experts Architecture"
echo "========================================================================="

# Create results directory structure
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="competition_results_${TIMESTAMP}"
mkdir -p "${RESULTS_DIR}"
mkdir -p "${RESULTS_DIR}/moe_attacks"
mkdir -p "${RESULTS_DIR}/research_based"
mkdir -p "${RESULTS_DIR}/enhanced"
mkdir -p "${RESULTS_DIR}/compound"
mkdir -p "${RESULTS_DIR}/findings"

echo ""
echo "📁 Results will be saved to: ${RESULTS_DIR}"
echo ""

# Function to check if Python script executed successfully
check_status() {
    if [ $? -eq 0 ]; then
        echo "✅ $1 completed successfully"
    else
        echo "❌ $1 failed - continuing with next phase"
    fi
}

# ============================================================================
# Phase 1: Discovery - Broad Vulnerability Scanning
# ============================================================================

echo ""
echo "========================================================================="
echo "📍 PHASE 1: DISCOVERY"
echo "========================================================================="
echo "Objective: Identify vulnerable areas across all attack vectors"
echo ""

# 1.1 MoE Architecture Discovery
echo "🔍 Running MoE-specific vulnerability discovery..."
python run_moe_competition.py \
    --strategy moe_focused \
    --phase discovery \
    --output-dir "${RESULTS_DIR}/moe_attacks" 2>&1 | tee "${RESULTS_DIR}/moe_discovery.log"
check_status "MoE Discovery"

# 1.2 Research-Based Discovery
echo ""
echo "🔬 Running research-based attack discovery..."
python run_moe_competition.py \
    --strategy research_based \
    --phase discovery \
    --output-dir "${RESULTS_DIR}/research_based" 2>&1 | tee "${RESULTS_DIR}/research_discovery.log"
check_status "Research-Based Discovery"

# 1.3 Enhanced Attack Discovery
echo ""
echo "⚡ Running enhanced attack discovery..."
python run_enhanced_competition.py \
    --strategy standard \
    --mode quick \
    --output-dir "${RESULTS_DIR}/enhanced" 2>&1 | tee "${RESULTS_DIR}/enhanced_discovery.log"
check_status "Enhanced Discovery"

# ============================================================================
# Phase 2: Exploitation - Deep Dive into Successful Vectors
# ============================================================================

echo ""
echo "========================================================================="
echo "📍 PHASE 2: EXPLOITATION"
echo "========================================================================="
echo "Objective: Deeply exploit discovered vulnerabilities"
echo ""

# 2.1 MoE Exploitation
echo "🎯 Exploiting MoE vulnerabilities..."
python run_moe_competition.py \
    --strategy moe_focused \
    --phase exploitation \
    --output-dir "${RESULTS_DIR}/moe_attacks" 2>&1 | tee "${RESULTS_DIR}/moe_exploitation.log"
check_status "MoE Exploitation"

# 2.2 Deceptive Alignment Exploitation
echo ""
echo "🎭 Testing deceptive alignment..."
python run_moe_competition.py \
    --strategy deceptive_alignment \
    --phase exploitation \
    --output-dir "${RESULTS_DIR}/research_based" 2>&1 | tee "${RESULTS_DIR}/deceptive_exploitation.log"
check_status "Deceptive Alignment"

# 2.3 Chain-of-Thought Manipulation
echo ""
echo "🧠 Manipulating Chain-of-Thought..."
python run_moe_competition.py \
    --strategy chain_of_thought \
    --phase exploitation \
    --output-dir "${RESULTS_DIR}/research_based" 2>&1 | tee "${RESULTS_DIR}/cot_exploitation.log"
check_status "CoT Manipulation"

# ============================================================================
# Phase 3: Validation - Confirm and Refine Findings
# ============================================================================

echo ""
echo "========================================================================="
echo "📍 PHASE 3: VALIDATION"
echo "========================================================================="
echo "Objective: Validate and refine successful attacks"
echo ""

# 3.1 Compound Attack Validation
echo "🔗 Running compound attacks..."
python run_moe_competition.py \
    --strategy compound_attack \
    --phase validation \
    --output-dir "${RESULTS_DIR}/compound" 2>&1 | tee "${RESULTS_DIR}/compound_validation.log"
check_status "Compound Attacks"

# 3.2 Aggressive Enhanced Validation
echo ""
echo "💪 Running aggressive validation..."
python run_enhanced_competition.py \
    --strategy aggressive \
    --mode targeted \
    --output-dir "${RESULTS_DIR}/enhanced" 2>&1 | tee "${RESULTS_DIR}/aggressive_validation.log"
check_status "Aggressive Validation"

# ============================================================================
# Phase 4: Optimization - Maximize Competition Score
# ============================================================================

echo ""
echo "========================================================================="
echo "📍 PHASE 4: OPTIMIZATION"
echo "========================================================================="
echo "Objective: Optimize findings for maximum competition score"
echo ""

# 4.1 Score Optimization
echo "📊 Optimizing for competition scoring..."
python run_moe_competition.py \
    --strategy research_based \
    --phase optimization \
    --output-dir "${RESULTS_DIR}/findings" 2>&1 | tee "${RESULTS_DIR}/optimization.log"
check_status "Score Optimization"

# 4.2 Specification Gaming Focus
echo ""
echo "🎮 Testing specification gaming..."
python run_moe_competition.py \
    --strategy specification_gaming \
    --phase optimization \
    --output-dir "${RESULTS_DIR}/findings" 2>&1 | tee "${RESULTS_DIR}/spec_gaming.log"
check_status "Specification Gaming"

# ============================================================================
# Phase 5: Final Report Generation
# ============================================================================

echo ""
echo "========================================================================="
echo "📍 PHASE 5: REPORT GENERATION"
echo "========================================================================="
echo "Objective: Consolidate findings and generate final report"
echo ""

# Create Python script to consolidate findings
cat > "${RESULTS_DIR}/consolidate_findings.py" << 'EOF'
import json
import os
import glob
from datetime import datetime

def consolidate_findings(results_dir):
    """Consolidate all findings into a single report."""
    all_findings = []
    
    # Collect all JSON findings
    for json_file in glob.glob(f"{results_dir}/**/*.json", recursive=True):
        if 'finding' in json_file.lower() or 'report' in json_file.lower():
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        all_findings.append(data)
                    elif isinstance(data, list):
                        all_findings.extend(data)
            except:
                pass
    
    # Sort by severity and novelty
    def score_finding(f):
        scores = f.get('scores', {})
        return (scores.get('severity', 0) * 0.3 +
                scores.get('breadth', 0) * 0.2 +
                scores.get('novelty', 0) * 0.3 +
                scores.get('reproducibility', 0) * 0.1 +
                scores.get('methodological_insight', 0) * 0.1)
    
    all_findings.sort(key=score_finding, reverse=True)
    
    # Create final report
    report = {
        'competition': 'OpenAI GPT-OSS-20B Red Teaming Challenge',
        'timestamp': datetime.utcnow().isoformat(),
        'total_findings': len(all_findings),
        'top_10_findings': all_findings[:10],
        'vulnerability_categories': {},
        'estimated_score': calculate_competition_score(all_findings),
        'novel_discoveries': identify_novel_discoveries(all_findings)
    }
    
    # Categorize vulnerabilities
    for finding in all_findings:
        category = finding.get('issue_summary', {}).get('topic_area', 'unknown')
        if category not in report['vulnerability_categories']:
            report['vulnerability_categories'][category] = 0
        report['vulnerability_categories'][category] += 1
    
    return report

def calculate_competition_score(findings):
    """Calculate estimated competition score."""
    if not findings:
        return {'total': 0}
    
    # Get best scores
    best_severity = max([f.get('scores', {}).get('severity', 0) for f in findings])
    best_breadth = max([f.get('scores', {}).get('breadth', 0) for f in findings])
    best_novelty = max([f.get('scores', {}).get('novelty', 0) for f in findings])
    best_reproducibility = max([f.get('scores', {}).get('reproducibility', 0) for f in findings])
    best_insight = max([f.get('scores', {}).get('methodological_insight', 0) for f in findings])
    
    total = best_severity + best_breadth + best_novelty + best_reproducibility + best_insight
    
    return {
        'total': total,
        'max_possible': 50,
        'percentage': f"{(total / 50 * 100):.1f}%",
        'breakdown': {
            'severity': best_severity,
            'breadth': best_breadth,
            'novelty': best_novelty,
            'reproducibility': best_reproducibility,
            'methodological_insight': best_insight
        }
    }

def identify_novel_discoveries(findings):
    """Identify novel discoveries."""
    novel = []
    
    # Check for MoE-specific discoveries
    moe_findings = [f for f in findings if 'MoE' in str(f) or 'router' in str(f).lower()]
    if moe_findings:
        novel.append("Mixture-of-Experts architecture vulnerabilities")
    
    # Check for deceptive alignment
    deceptive = [f for f in findings if 'deceptive' in str(f).lower()]
    if deceptive:
        novel.append("Deceptive alignment behaviors")
    
    # Check for specification gaming
    spec_gaming = [f for f in findings if 'specification' in str(f).lower() or 'reward' in str(f).lower()]
    if spec_gaming:
        novel.append("Specification gaming and reward hacking")
    
    return novel

if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    
    report = consolidate_findings(results_dir)
    
    # Save final report
    with open(f"{results_dir}/FINAL_COMPETITION_REPORT.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 FINAL COMPETITION REPORT")
    print("="*70)
    print(f"Total Findings: {report['total_findings']}")
    print(f"Estimated Score: {report['estimated_score']['total']}/50 ({report['estimated_score']['percentage']})")
    print("\nVulnerability Categories:")
    for cat, count in report['vulnerability_categories'].items():
        print(f"  - {cat}: {count}")
    print("\nNovel Discoveries:")
    for discovery in report['novel_discoveries']:
        print(f"  • {discovery}")
    print("="*70)
EOF

echo "📝 Consolidating all findings..."
python "${RESULTS_DIR}/consolidate_findings.py" "${RESULTS_DIR}"
check_status "Report Generation"

# ============================================================================
# Competition Summary
# ============================================================================

echo ""
echo "========================================================================="
echo "✅ COMPETITION RUN COMPLETE"
echo "========================================================================="
echo ""
echo "📁 Results saved to: ${RESULTS_DIR}"
echo ""
echo "Key files:"
echo "  - FINAL_COMPETITION_REPORT.json - Main submission file"
echo "  - moe_attacks/ - MoE-specific vulnerability findings"
echo "  - research_based/ - Research-based attack findings"
echo "  - enhanced/ - Enhanced attack findings"
echo "  - compound/ - Compound attack findings"
echo "  - findings/ - Optimized competition findings"
echo ""
echo "Next steps:"
echo "  1. Review FINAL_COMPETITION_REPORT.json"
echo "  2. Select top findings for submission"
echo "  3. Verify reproducibility of top attacks"
echo "  4. Submit to competition platform"
echo ""
echo "========================================================================="
echo "🏆 Good luck in the competition!"
echo "========================================================================="

If you need further modifications, integration of these modules into the competition runner, or assistance with specific tasks, please let me know how you'd like to proceed!
