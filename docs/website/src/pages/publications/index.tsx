// A component to read a bibliography from a stored bibtex file and render it as
// pretty HTML
//
// TODO: Write a small React component to parse the bibtex, and render it as a list

import React, {useState, useEffect, useCallback} from 'react';

import Layout from '@theme/Layout';
import * as path from 'path';
import bibtexParse from '@orcid/bibtex-parse-js';

function renderEntries(entries) {

    var citationKeys = []
    var titles = []
    var abstracts = []
    var bibtex = []
    var resource = []
    var years = []

    entries.forEach((entry) => {
        citationKeys.push(entry.citationKey);
        titles.push(entry.entryTags.title);
        years.push(entry.entryTags.year);
        abstracts.push(entry.entryTags.abstract);
    });


    return (
 <Layout title="Bibliography" description="Bibliography of CaBRNet">
      <div
        style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '50vh',
        }}>


        <div>{titles[0]}, {abstracts[0]}, {years[0]}</div>
        </div>
    </Layout>
    )
}

const Bibtex  = () => {
    const [entries, setEntries] = useState(null);
    const [loaded, setLoaded] = useState(null);

    useEffect(() => {
        const fetchBib = async () => {
            try {
                setLoaded(false);
                const response = await fetch("/bibliography/publications.bib");
                const text = await response.text()
                const bib = bibtexParse.toJSON(text)
                setEntries(bib);
                setLoaded(true);
            } catch(err) {
                console.log(err);
            };
        };
        fetchBib();
    }, []);

    console.log(entries)


    if (loaded) return(<div>{renderEntries(entries)}</div>);

}


export default Bibtex;
