import BridgeStatusChip from "./BridgeStatusChip";

interface BrandProps {
  onReset?: () => void;
}

export default function Brand({ onReset }: BrandProps) {
  return (
    <div className="flex items-center justify-between">
      <button
        className="group flex items-center gap-3 text-left"
        onClick={onReset}
        disabled={!onReset}
      >
        <Mark />
        <div>
          <div className="font-display text-xl tracking-tight text-ink-100">
            SwingSage
          </div>
          <div className="label-eyebrow">Local golf AI</div>
        </div>
      </button>

      <div className="hidden items-center gap-3 md:flex">
        <BridgeStatusChip />
        <span className="label-eyebrow">Runtime</span>
        <Pill label="RTX 5080" />
        <Pill label="Qwen 3 14B" />
      </div>
    </div>
  );
}

function Mark() {
  return (
    <div className="grid h-11 w-11 place-items-center rounded-xl bg-ink-800 hairline transition-shadow group-hover:shadow-glow">
      <svg viewBox="0 0 40 40" className="h-7 w-7">
        <path
          d="M12 28 L28 12"
          stroke="#E6C488"
          strokeWidth="2.5"
          strokeLinecap="round"
        />
        <circle cx="27" cy="13" r="3.2" fill="#E6C488" />
        <circle cx="13" cy="27" r="1.6" fill="#6EE7A1" />
      </svg>
    </div>
  );
}

function Pill({ label }: { label: string }) {
  return (
    <span className="rounded-full hairline bg-ink-800/60 px-3 py-1 font-mono text-[10px] uppercase tracking-widest text-ink-200">
      {label}
    </span>
  );
}
