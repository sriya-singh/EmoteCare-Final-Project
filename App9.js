


  document.addEventListener('DOMContentLoaded', function () {
    const appContainer = document.getElementById('app');

    function toggleSubsections(section) {
      section.classList.toggle('clicked');
      const subsections = section.querySelector('.subsections');
      subsections.style.display = subsections.style.display === 'block' ? 'none' : 'block';
    }

    function createSubsection(title, content, helplineNumber, websiteLink) {
      const subsection = document.createElement('div');
      subsection.className = 'subsection';

      const heading = document.createElement('h3');
      heading.textContent = title;

      const paragraph = document.createElement('p');
      paragraph.textContent = content;

      const helpline = document.createElement('p');
      helpline.textContent = `Helpline: ${helplineNumber}`;

      const anchor = document.createElement('a');
      anchor.href = websiteLink;
      anchor.target = '_blank';
      anchor.rel = 'noopener noreferrer';
      anchor.textContent = 'Website Link';

      subsection.appendChild(heading);
      subsection.appendChild(paragraph);
      subsection.appendChild(helpline);
      subsection.appendChild(anchor);

      return subsection;
    }

    function createSection(title, id, subsections) {
      const section = document.createElement('section');
      section.className = 'section';
      section.id = id;

      const heading = document.createElement('h2');
      heading.textContent = title;

      const subsectionContainer = document.createElement('div');
      subsectionContainer.className = 'subsections';

      subsections.forEach((subsection) => {
        subsectionContainer.appendChild(subsection);
      });

      section.appendChild(heading);
      section.appendChild(subsectionContainer);

      section.addEventListener('click', () => toggleSubsections(section));

      return section;
    }

    function renderTherapyPage() {
      const therapyPage = document.createElement('div');
      therapyPage.className = 'container';

      const title = document.createElement('h1');
      title.textContent = 'Therapy';

      const therapySection = createSection(
        'Therapy',
        'therapy',
        [
          createSubsection('Individual Therapy', 'Description for Individual Therapy', '', ''),
          createSubsection('Couple Therapy', 'Description for Couple Therapy', '', ''),
          createSubsection('Family Therapy', 'Description for Family Therapy', '', '')
        ]
      );

      therapyPage.appendChild(title);
      therapyPage.appendChild(therapySection);

      appContainer.appendChild(therapyPage);
    }

    renderTherapyPage();
  });

