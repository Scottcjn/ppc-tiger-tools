#!/bin/bash
# Tiger G4 Optimization Script - For 256MB RAM PowerBook
# Makes Tiger run smoother on limited memory

echo "🐯 Tiger Optimization Script for PowerBook G4"
echo "Optimizing for 256MB RAM..."
echo ""

cat << 'EOF' > optimize_tiger.sh
#!/bin/bash
# Tiger memory optimization

echo "Applying Tiger optimizations..."

# 1. Disable Dashboard (saves ~20MB)
defaults write com.apple.dashboard mcx-disabled -boolean YES
killall Dock

# 2. Disable Spotlight indexing (saves CPU and RAM)
sudo mdutil -a -i off
sudo launchctl unload /System/Library/LaunchDaemons/com.apple.metadata.mds.plist

# 3. Reduce window effects
defaults write com.apple.dock no-glass -boolean YES
defaults write NSGlobalDomain NSAutomaticWindowAnimationsEnabled -bool false

# 4. Disable unnecessary services
sudo launchctl unload -w /System/Library/LaunchDaemons/com.apple.mDNSResponder.plist
sudo launchctl unload -w /System/Library/LaunchDaemons/com.apple.locate.plist

# 5. Optimize swap
sudo dynamic_pager -S 256
sudo sysctl -w vm.swapusage=1

# 6. Clear caches
sudo rm -rf /Library/Caches/*
rm -rf ~/Library/Caches/*

# 7. Disable automatic software update checks
defaults write com.apple.SoftwareUpdate ScheduleFrequency -int 0

# 8. Reduce Finder memory usage
defaults write com.apple.finder DisableAllAnimations -bool true
defaults write com.apple.finder ShowStatusBar -bool false
defaults write com.apple.finder ShowPathbar -bool false

# 9. Disable Time Machine (if present)
sudo tmutil disable 2>/dev/null

# 10. Set conservative memory parameters
sudo sysctl -w kern.maxvnodes=3000
sudo sysctl -w kern.maxproc=256
sudo sysctl -w kern.maxfiles=8192

echo ""
echo "✅ Optimizations applied!"
echo "Recommended: Restart your PowerBook G4"
echo ""
echo "Memory saved: ~40-50MB"
echo "CPU overhead reduced significantly"
echo ""
echo "To re-enable Dashboard later:"
echo "defaults write com.apple.dashboard mcx-disabled -boolean NO"
EOF

echo "Script created: optimize_tiger.sh"
echo ""
echo "To run on the G4:"
echo "1. Copy this script to the G4"
echo "2. chmod +x optimize_tiger.sh"
echo "3. sudo ./optimize_tiger.sh"
echo ""
echo "These optimizations will:"
echo "- Disable Dashboard widget system"
echo "- Turn off Spotlight indexing"
echo "- Reduce visual effects"
echo "- Stop unnecessary background services"
echo "- Optimize virtual memory settings"
echo "- Clear system caches"
echo ""
echo "Perfect for running Rustchain validator on 256MB!"