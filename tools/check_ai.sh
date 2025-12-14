#!/bin/bash
echo "=== AI RECOMMENDATIONS ==="
grep -i "recommend\|suggest\|stack\|avoid\|prefer" test_output.log | head -20

echo -e "\n=== FINAL LINEUPS ==="
grep "H2H MVP:" test_output.log

echo -e "\n=== SALARY USAGE ==="
grep "total_salary" test_output.log | tail -3
