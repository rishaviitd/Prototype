bash <<'EOF'
##############################################################################
#  HEIC ➜ PNG ➜ JPEG (200–300 kB)  for tt-1 … tt-5   •  macOS-only (uses sips)
##############################################################################

lower=200000           # 200 kB in bytes
upper=307200           # 300 kB in bytes
min_side=800           # don’t shrink longest edge below 800 px

for n in 1 2 3 4 5; do
  heic=""
  for ext in HEIC heic; do                     # handle either case
    [[ -f "tt-${n}.${ext}" ]] && { heic="tt-${n}.${ext}"; break; }
  done
  if [[ -z $heic ]]; then
    echo "⚠︎  tt-${n}.HEIC not found — skipping"
    continue
  fi

  png="tt-${n}.png"
  jpg="tt-${n}.jpg"

  echo "▶︎  $heic → $png"
  if ! sips -s format png "$heic" --out "$png" >/dev/null; then
    echo "❌  PNG conversion failed for $heic"
    continue
  fi

  # --------  Step 1: JPEG at gradually lower quality  --------
  quality=80
  sips -s format jpeg -s formatOptions "$quality" "$png" --out "$jpg" >/dev/null
  size=$(stat -f%z "$jpg")
  while (( size > upper && quality > 10 )); do
    quality=$(( quality - 10 ))
    sips -s format jpeg -s formatOptions "$quality" "$png" --out "$jpg" >/dev/null
    size=$(stat -f%z "$jpg")
  done

  # --------  Step 2: down-scale resolution if still too big  --------
  if (( size > upper )); then
    longest=$(sips -g pixelWidth -g pixelHeight "$jpg" |
              awk '/pixel(W|H)eight/ {print $2}' | sort -nr | head -1)
    while (( size > upper && longest > min_side )); do
      longest=$(( longest * 80 / 100 ))             # shrink to 80 %
      sips -Z "$longest" "$jpg" >/dev/null          # keep aspect ratio
      size=$(stat -f%z "$jpg")
    done
  fi

  # --------  Final decision  --------
  if (( size >= lower && size <= upper )); then
    printf "✓  %s → %s  (%0.1f kB, q=%d)\n" \
           "$heic" "$jpg" "$(bc -l <<<"$size/1024")" "$quality"
    rm "$heic" "$png"
  else
    echo "❌  Couldn’t hit 200–300 kB without overshrinking — keeping originals"
    rm -f "$jpg"
  fi
done
EOF

# Heic to png conversion ----------------------------------------------------->

bash <<'EOF'
#!/usr/bin/env bash
##############################################################################
#  HEIC ➜ PNG  (macOS-only — uses “sips”)
##############################################################################

shopt -s nullglob nocaseglob            # allow empty globs + case-insensitive
for img in *.heic; do
  png="${img%.*}.png"
  echo "▶︎  $img → $png"
  if sips -s format png "$img" --out "$png" >/dev/null; then
    echo "✓  Converted $img to $png"
  else
    echo "❌  Conversion failed for $img"
  fi
done
EOF


# Png to smaller png compression ----------------------------------------------------->

bash <<'EOF'
##############################################################################
#  PNG ➜ smaller PNG (300–400 kB)  •  macOS-only tools: pngquant + sips
#  - Needs:  pngquant  (brew install pngquant)
##############################################################################

lower=307200     # 300 kB in bytes
upper=409600     # 400 kB in bytes
min_side=800     # don’t shrink longest edge below 800 px

shopt -s nullglob nocaseglob             # empty globs & case-insensitive *.png
for png in *.png; do
  echo "▶︎  Processing $png"
  tmp="$(mktemp -u "${png%.*}-XXXX.png")"   # temp working file

  # --------  Step 1: palette compression via pngquant  --------
  quality=90                                # start high, drop if still too big
  cp "$png" "$tmp"
  pngquant --force --quality=$((quality-5))-"$quality" --output "$tmp" "$tmp"
  size=$(stat -f%z "$tmp")                  # file size in bytes

  while (( size > upper && quality > 30 )); do
    quality=$(( quality - 10 ))
    cp "$png" "$tmp"
    pngquant --force --quality=$((quality-5))-"$quality" --output "$tmp" "$tmp"
    size=$(stat -f%z "$tmp")
  done

  # --------  Step 2: down-scale resolution if still too big  --------
  if (( size > upper )); then
    longest=$(sips -g pixelWidth -g pixelHeight "$tmp" |
              awk '/pixel(W|H)eight/ {print $2}' | sort -nr | head -1)
    while (( size > upper && longest > min_side )); do
      longest=$(( longest * 85 / 100 ))      # shrink to 85 %
      sips -Z "$longest" "$tmp" >/dev/null   # keep aspect ratio
      size=$(stat -f%z "$tmp")
    done
  fi

  # --------  Final decision  --------
  if (( size >= lower && size <= upper )); then
    mv "$tmp" "$png"
    printf "✓  %s compressed to %0.1f kB (q≈%d)\n" \
           "$png" "$(bc -l <<<"$size/1024")" "$quality"
  else
    rm -f "$tmp"
    echo "❌  Couldn’t hit 300–400 kB without overshrinking — left original"
  fi
done
EOF
