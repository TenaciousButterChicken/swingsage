import { useCallback, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import Brand from "./components/Brand";
import UploadView from "./components/UploadView";
import ProcessingView from "./components/ProcessingView";
import ResultsView from "./components/ResultsView";
import type { AnalysisResult, StageName } from "./lib/types";

type Phase =
  | { kind: "upload" }
  | { kind: "processing"; jobId: string; filename: string }
  | { kind: "results"; result: AnalysisResult }
  | { kind: "error"; message: string };

const pageTransition = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -8 },
  transition: { duration: 0.35, ease: [0.22, 1, 0.36, 1] },
};

export default function App() {
  const [phase, setPhase] = useState<Phase>({ kind: "upload" });

  const goUpload = useCallback(() => setPhase({ kind: "upload" }), []);

  const onJobStarted = useCallback((jobId: string, filename: string) => {
    setPhase({ kind: "processing", jobId, filename });
  }, []);

  const onJobDone = useCallback((result: AnalysisResult) => {
    setPhase({ kind: "results", result });
  }, []);

  const onJobError = useCallback((message: string) => {
    setPhase({ kind: "error", message });
  }, []);

  return (
    <div className="relative z-10 flex min-h-screen flex-col">
      <header className="px-8 pt-8 md:px-14 md:pt-10">
        <Brand onReset={phase.kind !== "upload" ? goUpload : undefined} />
      </header>

      <main className="flex flex-1 items-start justify-center px-6 pb-16 pt-8 md:px-14">
        <div className="w-full max-w-6xl">
          <AnimatePresence mode="wait">
            {phase.kind === "upload" && (
              <motion.div key="upload" {...pageTransition}>
                <UploadView onJobStarted={onJobStarted} onError={onJobError} />
              </motion.div>
            )}
            {phase.kind === "processing" && (
              <motion.div key="processing" {...pageTransition}>
                <ProcessingView
                  jobId={phase.jobId}
                  filename={phase.filename}
                  onDone={onJobDone}
                  onError={onJobError}
                />
              </motion.div>
            )}
            {phase.kind === "results" && (
              <motion.div key="results" {...pageTransition}>
                <ResultsView result={phase.result} onReset={goUpload} />
              </motion.div>
            )}
            {phase.kind === "error" && (
              <motion.div key="error" {...pageTransition}>
                <ErrorCard message={phase.message} onReset={goUpload} />
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>

      <footer className="px-8 pb-6 text-center md:px-14">
        <p className="label-eyebrow">SwingSage · local · no cloud</p>
      </footer>
    </div>
  );
}

function ErrorCard({ message, onReset }: { message: string; onReset: () => void }) {
  return (
    <div className="card-glass mx-auto max-w-2xl p-10">
      <p className="label-eyebrow mb-3 text-ember-400">Something went wrong</p>
      <pre className="whitespace-pre-wrap font-mono text-xs text-ink-200">{message}</pre>
      <div className="mt-8">
        <button className="btn-ghost" onClick={onReset}>
          Start over
        </button>
      </div>
    </div>
  );
}

// Re-export a stage-name utility for any downstream component that wants it.
export type { StageName };
