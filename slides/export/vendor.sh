#!/bin/sh
# Copy reveal.js and KaTeX into deck/vendor so the deck runs offline.
set -e
cd "$(dirname "$0")/.."
rm -rf deck/vendor && mkdir -p deck/vendor/reveal deck/vendor/katex/dist
cp -r node_modules/reveal.js/dist/. deck/vendor/reveal/
cp -r node_modules/reveal.js/plugin deck/vendor/reveal/plugin
cp -r node_modules/katex/dist/. deck/vendor/katex/dist/
echo "vendored reveal.js $(node -p "require('./node_modules/reveal.js/package.json').version"), katex $(node -p "require('./node_modules/katex/package.json').version")"
mkdir -p deck/vendor/fonts
cp -r node_modules/@fontsource/inter/files deck/vendor/fonts/inter
cp node_modules/@fontsource/inter/latin-400.css node_modules/@fontsource/inter/latin-500.css node_modules/@fontsource/inter/latin-600.css deck/vendor/fonts/
cp -r node_modules/@fontsource/jetbrains-mono/files deck/vendor/fonts/jetbrains-mono
cp node_modules/@fontsource/jetbrains-mono/latin-400.css deck/vendor/fonts/jbm-latin-400.css
sed -i 's#./files/#./inter/#g' deck/vendor/fonts/latin-*.css
sed -i 's#./files/#./jetbrains-mono/#g' deck/vendor/fonts/jbm-latin-400.css
