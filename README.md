# AMV Director 🎬

An experimental AI-assisted anime music video director that learns editing language from reference AMVs and applies it to the user's own anime footage.

## Vision

Given:

- a music track
- raw anime footage
- a reference AMV

AMV Director aims to automatically create a new edit that follows the reference's editing language while making original creative decisions using the user's footage.

The reference is treated as an **editing teacher**, not as footage to copy.

## Architecture

```text
Reference AMV
      +
Music
      +
Source Anime Footage
      ↓
Reference Analysis
      ↓
Music Director
      ↓
Audiovisual Event Analysis
      ↓
Reference AV Grammar
      ↓
Music Lock
      ↓
Creative Director
      ↓
Effect Director
      ↓
Deterministic FFmpeg Renderer
      ↓
Critic
      ↓
Auto Revision