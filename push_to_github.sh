#!/bin/bash
# Script to push the project to GitHub

cd "/Users/vishnuvaibhav/Library/Mobile Documents/com~apple~CloudDocs/school/265_final"

echo "🚀 Pushing to GitHub..."
echo ""

# Check if remote is set
if ! git remote get-url origin > /dev/null 2>&1; then
    echo "❌ Remote not configured. Setting up..."
    git remote add origin https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-.git
fi

# Ensure we're on main branch
git branch -M main

# Push to GitHub
echo "📤 Pushing to origin/main..."
git push -u origin main

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Successfully pushed to GitHub!"
    echo "🌐 View your repository at:"
    echo "   https://github.com/bvishnu08/diabetes-readmission-prediction-with-flagging-hisk-risk-patiences-"
else
    echo ""
    echo "⚠️  Push failed. This usually means:"
    echo "   1. The repository doesn't exist on GitHub yet (create it first)"
    echo "   2. Authentication is required (you'll be prompted for credentials)"
    echo ""
    echo "💡 If you need to create the repository, go to:"
    echo "   https://github.com/new"
fi

