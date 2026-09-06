import { useEffect, useRef, type ReactNode } from 'react';

export function Modal({ title, onClose, children }: { title: string; onClose: () => void; children: ReactNode }) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => { ref.current?.showModal(); }, []);
  return <dialog ref={ref} className="setup-dialog" aria-label={title} onCancel={onClose}>
    <div className="setup-header"><h2>{title}</h2><button onClick={onClose} aria-label={title === 'Agent connections' ? 'Close connections' : 'Close dialog'}>Close</button></div>
    {children}
  </dialog>;
}
