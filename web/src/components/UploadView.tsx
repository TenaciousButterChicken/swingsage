import { useCallback, useRef, useState } from "react";
import { motion } from "framer-motion";
import { uploadVideo } from "../lib/api";

interface UploadViewProps {
  onJobStarted: (jobId: string, filename: string) => void;
  onError: (message: string) => void;
}

const ACCEPTED = ["video/mp4", "video/quicktime", "video/x-matroska", "video/webm"];

export default function UploadView({ onJobStarted, onError }: UploadViewProps) {
  const [file, setFile] = useState<File | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const pickFile = useCallback((f: File | null | undefined) => {
    if (!f) return;
    const ok =
      ACCEPTED.includes(f.type) ||
      /\.(mp4|mov|m4v|mkv|webm)$/i.test(f.name);
    if (!ok) {
      onError(`Unsupported file type: ${f.type || f.name}. Use .mp4, .mov, .mkv or .webm.`);
      return;
    }
    setFile(f);
  }, [onError]);

  const onDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      pickFile(e.dataTransfer.files?.[0]);
    },
    [pickFile],
  );

  const analyze = useCallback(async () => {
    if (!file) return;
    setUploading(true);
    try {
      const { job_id, filename } = await uploadVideo(file);
      onJobStarted(job_id, filename);
    } catch (e) {
      setUploading(false);
      onError((e as Error).message);
    }
  }, [file, onJobStarted, onError]);

  return (
    <div className="flex flex-col items-center text-center">
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
        className="mb-10 max-w-3xl"
      >
        <p className="label-eyebrow mb-5">Analyse a swing</p>
        <h1 className="font-display text-display-xl leading-[1.02] text-ink-100">
          A private coach,
          <br />
          <span className="italic text-champagne-300">built for your room.</span>
        </h1>
        <p className="mx-auto mt-7 max-w-xl text-base text-ink-200">
          Drop one of your swing clips. No uploads leave your machine. Your
          RTX 5080 runs every pose frame, every biomechanical measurement,
          and Qwen 3 writes the feedback in about a minute.
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1], delay: 0.05 }}
        className="w-full"
      >
        <label
          htmlFor="file-input"
          onDragOver={(e) => {
            e.preventDefault();
            setDragOver(true);
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={onDrop}
          className={[
            "relative mx-auto flex w-full max-w-2xl cursor-pointer flex-col items-center justify-center rounded-3xl bg-ink-900/60 px-10 py-16 text-ink-200 transition-all duration-300 ease-out hairline",
            dragOver
              ? "shadow-glow border-champagne-300/40"
              : "hover:bg-ink-800/60 hover:shadow-card",
          ].join(" ")}
        >
          <input
            ref={inputRef}
            id="file-input"
            type="file"
            accept="video/*"
            className="hidden"
            onChange={(e) => pickFile(e.target.files?.[0] ?? null)}
          />
          <DropIcon active={dragOver} />
          <p className="mt-5 font-display text-2xl tracking-tight text-ink-100">
            {file ? file.name : "Drop your swing here"}
          </p>
          <p className="mt-2 text-sm text-ink-300">
            {file
              ? `${(file.size / (1024 * 1024)).toFixed(1)} MB · ready to analyse`
              : "or tap to browse · .mp4, .mov, .mkv up to ~200 MB"}
          </p>
        </label>

        <div className="mt-10 flex items-center justify-center gap-3">
          <button
            className="btn-primary min-w-[220px]"
            disabled={!file || uploading}
            onClick={analyze}
          >
            {uploading ? (
              <>
                <Spinner /> Uploading…
              </>
            ) : (
              <>
                Analyse swing
                <svg width="16" height="16" viewBox="0 0 16 16">
                  <path
                    d="M3 8 H13 M9 4 L13 8 L9 12"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    fill="none"
                  />
                </svg>
              </>
            )}
          </button>
          {file && !uploading && (
            <button
              className="btn-ghost"
              onClick={() => {
                setFile(null);
                if (inputRef.current) inputRef.current.value = "";
              }}
            >
              Clear
            </button>
          )}
        </div>

        <div className="mt-14 grid gap-6 text-left md:grid-cols-3">
          <Feature
            eyebrow="Step one"
            title="Record"
            body="Face-on or down-the-line, 30 fps, landscape. Hold address for a beat."
          />
          <Feature
            eyebrow="Step two"
            title="Analyse"
            body="Auto-trims the swing, locks onto the real top-of-backswing by geometry."
          />
          <Feature
            eyebrow="Step three"
            title="Coach"
            body="Schema-constrained feedback from Qwen 3, grounded in the real numbers."
          />
        </div>
      </motion.div>
    </div>
  );
}

function Feature({
  eyebrow,
  title,
  body,
}: {
  eyebrow: string;
  title: string;
  body: string;
}) {
  return (
    <div className="card-glass p-6">
      <p className="label-eyebrow mb-2 text-champagne-300">{eyebrow}</p>
      <h3 className="font-display text-xl tracking-tight text-ink-100">{title}</h3>
      <p className="mt-2 text-sm leading-relaxed text-ink-200">{body}</p>
    </div>
  );
}

function DropIcon({ active }: { active: boolean }) {
  return (
    <div
      className={[
        "grid h-16 w-16 place-items-center rounded-2xl bg-ink-800/70 transition-all duration-300",
        active ? "scale-110 bg-champagne-300/15" : "",
      ].join(" ")}
    >
      <svg viewBox="0 0 24 24" width="28" height="28" fill="none">
        <path
          d="M12 4 V16 M7 11 L12 16 L17 11"
          stroke={active ? "#E6C488" : "#BDB8AC"}
          strokeWidth="1.6"
          strokeLinecap="round"
          strokeLinejoin="round"
        />
        <path
          d="M4 20 H20"
          stroke={active ? "#E6C488" : "#8F8A7C"}
          strokeWidth="1.6"
          strokeLinecap="round"
        />
      </svg>
    </div>
  );
}

function Spinner() {
  return (
    <svg
      className="animate-spin"
      width="16"
      height="16"
      viewBox="0 0 24 24"
      fill="none"
    >
      <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2" opacity="0.25" />
      <path d="M21 12a9 9 0 0 0-9-9" stroke="currentColor" strokeWidth="2" strokeLinecap="round" />
    </svg>
  );
}
