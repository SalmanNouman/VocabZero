import React, { useState } from "react";
import { Search, Trash2, Database } from "lucide-react";
import type { LexiconEntry } from "../types";

interface LexiconTableProps {
  lexiconData: LexiconEntry[];
  onDelete: (sourceTerm: string) => void;
  isLoading: boolean;
  embedded?: boolean;
}

export const LexiconTable: React.FC<LexiconTableProps> = ({
  lexiconData,
  onDelete,
  isLoading,
  embedded = false,
}) => {
  const [searchQuery, setSearchQuery] = useState("");

  const filteredData = lexiconData.filter(
    (item) =>
      item.source_term.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.target_term.toLowerCase().includes(searchQuery.toLowerCase()),
  );

  const tableContent = (
    <>
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h3 className="text-lg font-semibold tracking-tight text-foreground flex items-center gap-2">
            <Database className="h-5 w-5 text-accent-foreground" />
            Lexicon
          </h3>
          <p className="text-xs text-muted-foreground mt-1">
            Rohingya acoustic signatures mapped dynamically to English
            translations
          </p>
        </div>

        {/* Filter input */}
        <div className="relative">
          <Search className="absolute left-3 top-2.5 h-4 w-4 text-text-dim" />
          <input
            type="text"
            placeholder="Search lexicon..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9 pr-4 py-2 text-sm bg-input border border-border rounded-lg text-foreground placeholder-text-dim focus:outline-none focus:ring-1 focus:ring-ring w-full sm:w-60"
          />
        </div>
      </div>

      {/* Table container */}
      <div className="overflow-x-auto rounded-lg border border-border bg-surface-sunken">
        <table className="w-full text-left border-collapse">
          <thead>
            <tr className="border-b border-border text-xs font-semibold text-muted-foreground bg-surface-raised/50">
              <th className="p-3.5">Rohingya Acoustic Signature</th>
              <th className="p-3.5">English Translation</th>
              <th className="p-3.5">Confidence</th>
              <th className="p-3.5 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border text-sm">
            {isLoading ? (
              [0, 1, 2, 3].map((i) => (
                <tr key={`skeleton-${i}`}>
                  <td className="p-3.5">
                    <div className="h-3.5 rounded bg-surface-raised animate-pulse w-24" />
                  </td>
                  <td className="p-3.5">
                    <div className="h-3.5 rounded bg-surface-raised animate-pulse w-32" />
                  </td>
                  <td className="p-3.5">
                    <div className="h-3.5 rounded bg-surface-raised animate-pulse w-12" />
                  </td>
                  <td className="p-3.5">
                    <div className="h-3.5 rounded bg-surface-raised animate-pulse w-8 ml-auto" />
                  </td>
                </tr>
              ))
            ) : filteredData.length === 0 ? (
              <tr>
                <td colSpan={4} className="p-12 text-center">
                  <div className="flex flex-col items-center gap-3">
                    <Database className="h-8 w-8 text-text-dim" />
                    <p className="text-sm text-muted-foreground">
                      {searchQuery
                        ? "No matching entries found."
                        : "No entries in local lexicon database."}
                    </p>
                  </div>
                </td>
              </tr>
            ) : (
              filteredData.map((item) => (
                <tr
                  key={item.source_term}
                  className="hover:bg-surface-raised/30 transition-colors"
                >
                  <td className="p-3.5 font-mono text-xs text-accent-foreground">
                    {item.source_term}
                  </td>
                  <td className="p-3.5 font-medium text-foreground">
                    {item.target_term}
                  </td>
                  <td className="p-3.5">
                    <span className="text-xs bg-surface-raised text-muted-foreground px-2 py-0.5 rounded border border-border-strong font-mono tabular-nums">
                      {item.confidence !== undefined
                        ? item.confidence.toFixed(2)
                        : "1.00"}
                    </span>
                  </td>
                  <td className="p-3.5 text-right">
                    <button
                      onClick={() => onDelete(item.source_term)}
                      className="p-1.5 hover:bg-surface-raised text-muted-foreground hover:text-destructive rounded-md transition-colors cursor-pointer"
                      title="Delete lexicon mapping"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>
    </>
  );

  if (embedded) {
    return <div className="flex flex-col gap-4">{tableContent}</div>;
  }

  return (
    <div className="bg-card border border-border rounded-xl p-5 shadow-[var(--shadow-md)] flex flex-col gap-4">
      {tableContent}
    </div>
  );
};
