/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Warm near-black base. Not pitch black — slight warmth reads
        // as expensive rather than harsh.
        ink: {
          950: "#070706",
          900: "#0A0A09",
          800: "#121210",
          700: "#1A1A17",
          600: "#26251F",
          500: "#3A3831",
          400: "#5E5B50",
          300: "#8F8A7C",
          200: "#BDB8AC",
          100: "#E6E2D6",
        },
        // Champagne / warm gold accent — golf-appropriate, reads luxe.
        champagne: {
          50: "#FBF6EC",
          100: "#F4E9D1",
          200: "#ECD9AC",
          300: "#E6C488",
          400: "#D4A867",
          500: "#B28842",
          600: "#8A6A32",
        },
        // Success/performance green.
        fairway: {
          400: "#6EE7A1",
          500: "#22C55E",
          600: "#16A34A",
        },
        // Alert / out-of-range.
        ember: {
          400: "#F59E9E",
          500: "#EF5C5C",
          600: "#C74040",
        },
      },
      fontFamily: {
        sans: ['"Geist"', '"Inter"', "system-ui", "sans-serif"],
        mono: ['"Geist Mono"', '"JetBrains Mono"', "ui-monospace", "monospace"],
        display: ['"Fraunces"', '"Geist"', "serif"],
      },
      backgroundImage: {
        // Very subtle vignette for the base background.
        "ink-radial":
          "radial-gradient(ellipse at 50% 0%, rgba(230,196,136,0.05) 0%, rgba(7,7,6,0) 45%), radial-gradient(ellipse at 50% 100%, rgba(34,197,94,0.03) 0%, rgba(7,7,6,0) 50%)",
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(230,196,136,0.12), 0 8px 40px -12px rgba(230,196,136,0.25)",
        card: "0 1px 0 rgba(255,255,255,0.03) inset, 0 24px 48px -24px rgba(0,0,0,0.5)",
      },
      backdropBlur: {
        xs: "2px",
      },
      fontSize: {
        "display-xl": ["4.5rem", { lineHeight: "1.02", letterSpacing: "-0.03em" }],
        "display-lg": ["3rem", { lineHeight: "1.05", letterSpacing: "-0.025em" }],
      },
    },
  },
  plugins: [],
};
