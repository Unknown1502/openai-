#!/usr/bin/env python3
"""Test script to verify the automated testing framework fix"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.competition.automated_testing_framework import AutomatedTestingFramework
from src.competition.research_based_attacks import RESEARCH_BASED_ATTACKS
from src.competition.moe_specific_attacks import MOE_ATTACK_VECTORS
from src.competition.enhanced_attack_vectors import ENHANCED_COMPETITION_ATTACK_VECTORS

def test_framework_initialization():
    """Test that the framework can be initialized"""
    print("Testing framework initialization...")
    try:
        framework = AutomatedTestingFramework()
        print("✅ Framework initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Framework initialization failed: {e}")
        return False

def test_test_suite_generation():
    """Test that test suite can be generated without errors"""
    print("\nTesting test suite generation...")
    try:
        framework = AutomatedTestingFramework()
        test_suite = framework.generate_comprehensive_test_suite(
            num_variations=2,  # Small number for testing
            include_mutations=False  # Skip mutations for quick test
        )
        
        print(f"✅ Generated {len(test_suite)} test cases")
        
        # Check that each test has required fields
        for test in test_suite[:5]:  # Check first 5
            assert "id" in test, "Test missing 'id'"
            assert "source" in test, "Test missing 'source'"
            assert "attack" in test, "Test missing 'attack'"
            assert "prompt" in test, "Test missing 'prompt'"
            assert "category" in test, "Test missing 'category'"
            assert "variations" in test, "Test missing 'variations'"
            
            # Verify prompt is not empty
            assert test["prompt"], f"Test {test['id']} has empty prompt"
            
            print(f"  - Test {test['id']}: {test['source']} - {len(test['prompt'])} chars")
        
        print("✅ All test cases have required fields")
        return True
        
    except Exception as e:
        print(f"❌ Test suite generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_research_attacks():
    """Test that research-based attacks work correctly"""
    print("\nTesting research-based attacks...")
    try:
        # Check that research attacks have prompt_sequence
        for attack in RESEARCH_BASED_ATTACKS[:3]:
            assert hasattr(attack, 'prompt_sequence'), f"Attack {attack.name} missing prompt_sequence"
            assert attack.prompt_sequence, f"Attack {attack.name} has empty prompt_sequence"
            assert isinstance(attack.prompt_sequence, list), f"Attack {attack.name} prompt_sequence is not a list"
            
            # Test obfuscation method
            obfuscated = attack.generate_obfuscated_prompt(0)
            assert obfuscated, f"Attack {attack.name} generated empty obfuscated prompt"
            
            print(f"  - {attack.name}: {len(attack.prompt_sequence)} prompts")
        
        print("✅ Research-based attacks working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Research attacks test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_moe_attacks():
    """Test that MoE attacks work correctly"""
    print("\nTesting MoE attacks...")
    try:
        # Check that MoE attacks have prompt_template
        for attack in MOE_ATTACK_VECTORS[:3]:
            assert hasattr(attack, 'prompt_template'), f"Attack {attack.name} missing prompt_template"
            assert attack.prompt_template, f"Attack {attack.name} has empty prompt_template"
            
            print(f"  - {attack.name}: {len(attack.prompt_template)} chars")
        
        print("✅ MoE attacks working correctly")
        return True
        
    except Exception as e:
        print(f"❌ MoE attacks test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_enhanced_attacks():
    """Test that enhanced attacks work correctly"""
    print("\nTesting enhanced attacks...")
    try:
        # Check that enhanced attacks have prompt_template
        for scenario in ENHANCED_COMPETITION_ATTACK_VECTORS[:3]:
            assert hasattr(scenario, 'prompt_template'), f"Scenario {scenario.name} missing prompt_template"
            assert scenario.prompt_template, f"Scenario {scenario.name} has empty prompt_template"
            
            print(f"  - {scenario.name}: {len(scenario.prompt_template)} chars")
        
        print("✅ Enhanced attacks working correctly")
        return True
        
    except Exception as e:
        print(f"❌ Enhanced attacks test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("AUTOMATED TESTING FRAMEWORK FIX VERIFICATION")
    print("="*60)
    
    tests = [
        test_framework_initialization,
        test_research_attacks,
        test_moe_attacks,
        test_enhanced_attacks,
        test_test_suite_generation
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    if all(results):
        print("✅ ALL TESTS PASSED - Framework fix successful!")
        print("\nThe automated testing framework should now work correctly.")
        print("The issue with 'prompt_template' attribute has been resolved.")
    else:
        print("❌ SOME TESTS FAILED - Please review the errors above")
        failed_count = len([r for r in results if not r])
        print(f"\nFailed tests: {failed_count}/{len(tests)}")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
