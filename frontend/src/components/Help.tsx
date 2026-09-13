/**
 * Help, as a page rather than a popup.
 *
 * It used to be a slide-over so the guidance sat beside whatever you were
 * stuck on. That stopped being enough once there was a second audience: a
 * technical reader wants the architecture, not a panel's worth of it, and
 * neither reader wants to lose their place in a modal every time they follow
 * a cross-reference. A page with its own URL (`#/help`, `#/help/technical`)
 * can be bookmarked, linked to a teammate, and read section by section.
 *
 * The two tracks are deliberately separate components (`HelpUserGuide`,
 * `HelpTechnical`) rather than one file with a filter, because they have
 * different authors in spirit — one written for whoever runs this, one for
 * whoever maintains it — and mixing them would end up serving neither well.
 */

import { useRef } from 'react';
import { HelpUserGuide, USER_GUIDE_TOC } from './HelpUserGuide';
import { HelpTechnical, TECHNICAL_TOC } from './HelpTechnical';

export type HelpSectionName = 'user' | 'technical';

type Props = {
  section: HelpSectionName;
  onSection: (section: HelpSectionName) => void;
};

export function Help({ section, onSection }: Props) {
  const bodyRef = useRef<HTMLDivElement>(null);

  const jumpTo = (id: string) => {
    bodyRef.current?.querySelector<HTMLElement>(`#${id}`)?.scrollIntoView({
      behavior: 'smooth',
      block: 'start',
    });
  };

  const toc = section === 'user' ? USER_GUIDE_TOC : TECHNICAL_TOC;

  return (
    <div className="help-page">
      <div className="help-page-head">
        <div>
          <h2>Help &amp; documentation</h2>
          <p className="hint">
            {section === 'user'
              ? 'How to use TRACE — recording, running, and reading what comes back.'
              : 'How TRACE is built — for the team maintaining or extending it.'}
          </p>
        </div>
        <div className="tabs help-page-tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={section === 'user'}
            className={section === 'user' ? 'active' : ''}
            onClick={() => onSection('user')}
          >
            User guide
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={section === 'technical'}
            className={section === 'technical' ? 'active' : ''}
            onClick={() => onSection('technical')}
          >
            Technical &amp; architecture
          </button>
        </div>
      </div>

      <div className="help-layout">
        <nav className="help-toc" aria-label="Sections on this page">
          {toc.map((group) => (
            <div className="help-toc-group" key={group.label}>
              <div className="help-toc-label">{group.label}</div>
              {group.items.map((item) => (
                <button type="button" key={item.id} onClick={() => jumpTo(item.id)}>
                  {item.title}
                </button>
              ))}
            </div>
          ))}
        </nav>

        <div className="help-body" ref={bodyRef}>
          {section === 'user' ? <HelpUserGuide /> : <HelpTechnical />}
        </div>
      </div>
    </div>
  );
}
