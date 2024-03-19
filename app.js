document.addEventListener('DOMContentLoaded', function () {
    const appContainer = document.getElementById('app');

    const toggleSubsections = (section) => {
        section.classList.toggle('clicked');
        const subsections = section.querySelector('.subsections');
        subsections.style.display = subsections.style.display === 'block' ? 'none' : 'block';
    };

    const createSubsection = (title, content, helplineNumber, websiteLink) => {
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
    };

    const createSection = (title, id, subsections) => {
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
    };

    const renderAwarenessPage = () => {
        const awarenessPage = document.createElement('div');
        awarenessPage.className = 'container';

        // Updated Headline and Tagline
        const title = document.createElement('h1');
        title.textContent = 'Resources Hub';

        const quote = document.createElement('p');
        quote.className = 'quote';
        quote.textContent = '"Empower Your Mind. Transform Your Well-being."'

        const helplinesSection = createSection(
            'Helplines',
            'helplines',
            [
                createSubsection('NIMHANS Helpline', 'National Institute of Mental Health and Neurosciences (NIMHANS) helpline', '+918046110007', 'https://nimhans.ac.in/'),
                createSubsection('Sneha Foundation (Chennai)', 'Sneha Foundation (Chennai) helpline', '+914425300123', 'https://www.snehaindia.org/'),
                createSubsection('Sanjivini Society for Mental Health (Delhi)', 'Sanjivini Society for Mental Health (Delhi) helpline', '+911126512211', 'https://www.sanjivinisociety.org/'),
                createSubsection('Aasra (Mumbai)', 'Aasra (Mumbai) helpline', '+912227546669', 'https://www.aasra.info/'),
                createSubsection('Lifeline Foundation (Kolkata)', 'Lifeline Foundation (Kolkata) helpline', '+913324617261', 'https://www.lifelinefoundation.in/'),
                createSubsection('Roshni Trust (Bangalore)', 'Roshni Trust (Bangalore) helpline', '+918025722577', 'http://roshni-trust.org/'),
                createSubsection('iCall (National)', 'iCall (National) helpline', '080-65309111', 'https://www.icallhelpline.org/')
            ]
        );

        const governmentSchemesSection = createSection(
            'Government Schemes',
            'government-schemes',
            [
                createSubsection('National Mental Health Programme (NMHP)', 'Launched by the Ministry of Health and Family Welfare, NMHP aims to improve the accessibility and affordability of mental healthcare services across India.', '', 'https://nhm.gov.in/index1.php?lang=1&level=2&sublinkid=1043&lid=359'),
                createSubsection('District Mental Health Programme (DMHP)', 'A component of NMHP, DMHP focuses on providing mental healthcare services at the district level.', '', 'https://nhm.gov.in/index1.php?lang=1&level=2&sublinkid=1043&lid=359'),
                createSubsection('Mental Health Helpline 181', 'This toll-free helpline provides information and support on mental health issues.', '', ''),
                createSubsection('National Health Mission (NHM)', 'NHM includes a mental health component that provides funding for mental healthcare services in states.', '', 'https://nhm.gov.in/')
            ]
        );

        const otherResourcesSection = createSection(
            'Other Resources',
            'other-resources',
            [
                createSubsection('The Banyan', 'A mental health organization that provides comprehensive mental healthcare services, including therapy, counseling, and rehabilitation.', '', 'https://www.thebanyan.org/'),
                createSubsection('Sangath', 'A mental health organization that works to promote mental well-being and provide support to people with mental illness.', '', 'https://www.sangath.in/'),
                createSubsection('MindPeers', 'A peer support network for people with mental illness.', '', 'https://www.bestmediainfo.com/'),
                createSubsection('It\'s Okay to Talk', 'A campaign that aims to reduce stigma around mental illness.', '', 'https://www.seemescotland.org/'),
                createSubsection('The Inclusive Forum', 'A platform for people with lived experience of mental illness to share their stories and connect with others.', '', 'https://www.edf-feph.org/')
            ]
        );

        awarenessPage.appendChild(title);
        awarenessPage.appendChild(quote);
        awarenessPage.appendChild(helplinesSection);
        awarenessPage.appendChild(governmentSchemesSection);
        awarenessPage.appendChild(otherResourcesSection);

        appContainer.appendChild(awarenessPage);
    };

    renderAwarenessPage();
});